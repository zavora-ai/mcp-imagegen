//! mcp-imagegen binary: validate the manifest, run the health check, serve over stdio.

use adk_mcp_sdk::HealthCheck;
use mcp_imagegen::{MANIFEST_TOML, server::ImageGenServer};
use rmcp::{ServiceExt, transport::stdio};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // stdout is the MCP channel, so logs go to stderr.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?),
        )
        .init();

    let manifest = adk_mcp_sdk::ServerManifest::from_toml(MANIFEST_TOML)?;
    let errors = manifest.validate();
    if !errors.is_empty() {
        for e in &errors {
            tracing::error!("manifest: {e}");
        }
        anyhow::bail!("invalid mcp-server.toml ({} error(s))", errors.len());
    }

    // Loads config + registry only; no weights are touched at startup.
    let server = ImageGenServer::from_default_config()?;
    let health = server.check_health().await;
    if !health.healthy {
        tracing::error!(message = ?health.message, "Health check failed");
        std::process::exit(1);
    }
    tracing::info!(
        config_dir = %server.config().config_dir.display(),
        health = health.message.as_deref().unwrap_or(""),
        "configuration loaded"
    );

    tracing::info!(
        "{} v{} starting on stdio",
        manifest.display_name,
        manifest.version
    );
    // Install signal handlers before the handshake so a stop request always gets a clean shutdown.
    let run = {
        let server = server.clone();
        async move {
            let service = server.serve(stdio()).await?;
            service.waiting().await?;
            anyhow::Ok(())
        }
    };
    tokio::select! {
        result = run => result?,
        _ = tokio::signal::ctrl_c() => tracing::info!("Ctrl-C received"),
        _ = terminate_signal() => tracing::info!("termination signal received"),
    }
    // Don't leave a multi-GB sd-server behind.
    server.shutdown().await;
    // The stdin reader may still be blocked; exit rather than wait for the runtime to drain it.
    std::process::exit(0);
}

/// SIGTERM on Unix; Ctrl-Break / console close on Windows.
#[cfg(unix)]
async fn terminate_signal() {
    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(mut sig) => {
            sig.recv().await;
        }
        Err(_) => std::future::pending().await,
    }
}

#[cfg(windows)]
async fn terminate_signal() {
    match tokio::signal::windows::ctrl_break() {
        Ok(mut sig) => {
            sig.recv().await;
        }
        Err(_) => std::future::pending().await,
    }
}

#[cfg(not(any(unix, windows)))]
async fn terminate_signal() {
    std::future::pending().await
}
