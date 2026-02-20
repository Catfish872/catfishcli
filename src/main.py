import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
    stream=sys.stdout,
    force=True  # This will override any existing handlers, ensuring our config is used
)

from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from .gemini_routes import router as gemini_router
from .openai_routes import router as openai_router
from .auth import get_credentials, get_user_project_id, onboard_user, get_accounts_status_snapshot
from .google_api_client import get_formatted_stats, get_usage_stats_snapshot
from .dashboard_monitor import init_inmemory_log_handler, get_recent_logs, get_log_overview
from .config import DASHBOARD_TOKEN, CREDENTIAL_FILE

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Environment variables loaded from .env file")
except ImportError:
    logging.warning("python-dotenv not installed, .env file will not be loaded automatically")
except Exception as e:
    logging.warning(f"Could not load .env file: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# Add CORS middleware for preflight requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


def _verify_dashboard_token(request: Request):
    token = (os.getenv("DASHBOARD_TOKEN") or DASHBOARD_TOKEN or "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="DASHBOARD_TOKEN is not configured")

    provided_token = request.headers.get("x-dashboard-token") or request.query_params.get("token")
    if not provided_token or provided_token != token:
        raise HTTPException(status_code=401, detail="Invalid dashboard token")


_DASHBOARD_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #111827; color: #e5e7eb; }
    header { padding: 16px; background: #1f2937; border-bottom: 1px solid #374151; }
    main { padding: 16px; display: grid; grid-template-columns: 1fr; gap: 16px; }
    .card { background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 12px; }
    h1, h2 { margin: 0 0 10px 0; }
    .muted { color: #9ca3af; font-size: 12px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #374151; padding: 8px; font-size: 13px; text-align: left; vertical-align: top; }
    code, pre { white-space: pre-wrap; word-break: break-word; }
    .ok { color: #10b981; }
    .warn { color: #f59e0b; }
    .err { color: #ef4444; }
    .log-container { max-height: 420px; overflow-y: auto; border: 1px solid #374151; border-radius: 6px; }
    #logs-table td pre { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  </style>
</head>
<body>
  <header>
    <h1>运行监控仪表盘</h1>
    <div class="muted">日志近实时刷新：1 秒 | 统计/凭证刷新：5 秒 | 通过 URL 参数 token 或请求头 x-dashboard-token 鉴权</div>
  </header>
  <main>
    <section class="card">
      <h2>调用统计</h2>
      <div id="stats-summary" class="muted">加载中...</div>
      <table>
        <thead><tr><th>Project ID</th><th>Success</th><th>Fail</th><th>Total</th></tr></thead>
        <tbody id="stats-table"></tbody>
      </table>
    </section>

    <section class="card">
      <h2>凭证状态</h2>
      <div id="accounts-summary" class="muted">加载中...</div>
      <table>
        <thead><tr><th>Project ID</th><th>Access Token</th><th>Refresh Token</th><th>Expiry(UTC)</th><th>Expired</th><th>Onboarding</th></tr></thead>
        <tbody id="accounts-table"></tbody>
      </table>
    </section>

    <section class="card">
      <h2>完整日志（最近 1000 条中的最新部分）</h2>
      <div id="logs-summary" class="muted">加载中...</div>
      <div class="log-container" id="logs-container">
        <table>
          <thead><tr><th style="width:180px;">Time(UTC)</th><th style="width:80px;">Level</th><th style="width:160px;">Logger</th><th>Message</th></tr></thead>
          <tbody id="logs-table"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const urlToken = new URLSearchParams(window.location.search).get('token') || '';

    async function fetchJson(path) {
      const resp = await fetch(path, {
        headers: { 'x-dashboard-token': urlToken }
      });
      if (!resp.ok) throw new Error(await resp.text());
      return await resp.json();
    }

    function esc(v) {
      const div = document.createElement('div');
      div.innerText = String(v ?? '');
      return div.innerHTML;
    }

    function boolTag(v) {
      if (v === true) return '<span class="ok">true</span>';
      if (v === false) return '<span class="err">false</span>';
      return '<span class="warn">unknown</span>';
    }

    let refreshingMeta = false;
    let refreshingLogs = false;

    async function refreshMeta() {
      if (refreshingMeta) return;
      refreshingMeta = true;
      try {
        const [stats, accounts] = await Promise.all([
          fetchJson('/dashboard/api/stats'),
          fetchJson('/dashboard/api/accounts')
        ]);

        document.getElementById('stats-summary').innerText =
          `reset=${stats.last_reset_date} | success=${stats.total_success} | fail=${stats.total_fail} | total=${stats.total}`;

        document.getElementById('stats-table').innerHTML = (stats.accounts || []).map(x => `
          <tr>
            <td>${esc(x.project_id)}</td><td>${esc(x.success)}</td><td>${esc(x.fail)}</td><td>${esc(x.total)}</td>
          </tr>
        `).join('') || '<tr><td colspan="4" class="muted">暂无数据</td></tr>';

        document.getElementById('accounts-summary').innerText = `total_accounts=${accounts.total_accounts}`;
        document.getElementById('accounts-table').innerHTML = (accounts.accounts || []).map(x => `
          <tr>
            <td>${esc(x.project_id)}</td>
            <td>${boolTag(x.has_access_token)}</td>
            <td>${boolTag(x.has_refresh_token)}</td>
            <td>${esc(x.expiry || '')}</td>
            <td>${boolTag(x.is_expired)}</td>
            <td>${boolTag(x.onboarding_complete)}</td>
          </tr>
        `).join('') || '<tr><td colspan="6" class="muted">暂无数据</td></tr>';
      } catch (e) {
        document.getElementById('stats-summary').innerText = `统计加载失败: ${e.message}`;
      } finally {
        refreshingMeta = false;
      }
    }

    async function refreshLogs() {
      if (refreshingLogs) return;
      refreshingLogs = true;
      try {
        const logs = await fetchJson('/dashboard/api/logs?limit=200');

        const container = document.getElementById('logs-container');
        const nearBottom = (container.scrollHeight - container.scrollTop - container.clientHeight) < 24;

        document.getElementById('logs-summary').innerText =
          `buffer_size=${logs.overview.buffer_size} | buffered=${logs.overview.buffered} | showing=${(logs.logs || []).length}`;

        document.getElementById('logs-table').innerHTML = (logs.logs || []).map(x => `
          <tr>
            <td>${esc(x.timestamp)}</td>
            <td>${esc(x.level)}</td>
            <td>${esc(x.logger)}</td>
            <td><pre>${esc(x.message)}</pre></td>
          </tr>
        `).join('') || '<tr><td colspan="4" class="muted">暂无日志</td></tr>';

        if (nearBottom) {
          container.scrollTop = container.scrollHeight;
        }
      } catch (e) {
        document.getElementById('logs-summary').innerText = `日志加载失败: ${e.message}`;
      } finally {
        refreshingLogs = false;
      }
    }

    refreshMeta();
    refreshLogs();
    setInterval(refreshMeta, 5000);
    setInterval(refreshLogs, 1000);
  </script>
</body>
</html>
"""


@app.on_event("startup")
async def startup_event():
    try:
        init_inmemory_log_handler()
        logging.info("Starting Gemini proxy server...")

        env_creds_json = os.getenv("GEMINI_CREDENTIALS")
        creds_file_exists = os.path.exists(CREDENTIAL_FILE)

        if env_creds_json or creds_file_exists:
            try:
                # Try to load existing credentials without OAuth flow first
                creds = get_credentials(allow_oauth_flow=False)
                if creds:
                    try:
                        proj_id = get_user_project_id(creds)
                        if proj_id:
                            onboard_user(creds, proj_id)
                            logging.info(f"Successfully onboarded with project ID: {proj_id}")
                        logging.info("Gemini proxy server started successfully")
                        logging.info("Authentication required - Password: see .env file")
                    except Exception as e:
                        logging.error(f"Setup failed: {str(e)}")
                        logging.warning("Server started but may not function properly until setup issues are resolved.")
                else:
                    logging.warning("Credentials file exists but could not be loaded. Server started - authentication will be required on first request.")
            except Exception as e:
                logging.error(f"Credential loading error: {str(e)}")
                logging.warning("Server started but credentials need to be set up.")
        else:
            # No credentials found - prompt user to authenticate
            logging.info("No credentials found. Starting OAuth authentication flow...")
            try:
                creds = get_credentials(allow_oauth_flow=True)
                if creds:
                    try:
                        proj_id = get_user_project_id(creds)
                        if proj_id:
                            onboard_user(creds, proj_id)
                            logging.info(f"Successfully onboarded with project ID: {proj_id}")
                        logging.info("Gemini proxy server started successfully")
                    except Exception as e:
                        logging.error(f"Setup failed: {str(e)}")
                        logging.warning("Server started but may not function properly until setup issues are resolved.")
                else:
                    logging.error("Authentication failed. Server started but will not function until credentials are provided.")
            except Exception as e:
                logging.error(f"Authentication error: {str(e)}")
                logging.warning("Server started but authentication failed.")

        logging.info("Authentication required - Password: see .env file")

    except Exception as e:
        logging.error(f"Startup error: {str(e)}")
        logging.warning("Server may not function properly.")


@app.options("/{full_path:path}")
async def handle_preflight(request: Request, full_path: str):
    """Handle CORS preflight requests without authentication."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    _verify_dashboard_token(request)
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/dashboard/api/logs")
async def dashboard_logs(request: Request, limit: int = Query(300, ge=1, le=1000)):
    _verify_dashboard_token(request)
    return {
        "overview": get_log_overview(),
        "logs": get_recent_logs(limit=limit),
    }


@app.get("/dashboard/api/accounts")
async def dashboard_accounts(request: Request):
    _verify_dashboard_token(request)
    return get_accounts_status_snapshot()


@app.get("/dashboard/api/stats")
async def dashboard_stats(request: Request):
    _verify_dashboard_token(request)
    return get_usage_stats_snapshot()


# Root endpoint - no authentication required
@app.get("/")
async def root():
    """
    Root endpoint providing project information.
    No authentication required.
    """
    return {
        "name": "geminicli2api",
        "description": "OpenAI-compatible API proxy for Google's Gemini models via gemini-cli",
        "purpose": "Provides both OpenAI-compatible endpoints (/v1/chat/completions) and native Gemini API endpoints for accessing Google's Gemini models",
        "version": "1.0.0",
        "daily_usage_stats": get_formatted_stats(),  # 添加统计报告
        "endpoints": {
            "openai_compatible": {
                "chat_completions": "/v1/chat/completions",
                "models": "/v1/models"
            },
            "native_gemini": {
                "models": "/v1beta/models",
                "generate": "/v1beta/models/{model}/generateContent",
                "stream": "/v1beta/models/{model}/streamGenerateContent"
            },
            "dashboard": "/dashboard",
            "health": "/health"
        },
        "authentication": "Required for all endpoints except root and health",
        "repository": "https://github.com/user/geminicli2api"
    }


# Health check endpoint for Docker/Hugging Face
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "geminicli2api"}


app.include_router(openai_router)
app.include_router(gemini_router)
