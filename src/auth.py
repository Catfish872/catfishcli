import os
import json
import base64
import time
import logging
import itertools
import threading
from datetime import datetime, timezone

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBasic
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleAuthRequest

from .utils import get_user_agent, get_client_metadata
from .config import (
    CLIENT_ID, CLIENT_SECRET, SCOPES, CREDENTIAL_FILE,
    CODE_ASSIST_ENDPOINT, GEMINI_AUTH_PASSWORD
)

# --- Global State for Account Polling ---
ACCOUNTS = []
account_cycler = None
onboarding_complete_map = {}  # Track onboarding status per project_id
file_lock = threading.Lock()

security = HTTPBasic()

def _load_accounts():
    """Loads all accounts from the credential file and prepares the cycler."""
    global ACCOUNTS, account_cycler
    if not os.path.exists(CREDENTIAL_FILE):
        logging.warning(f"Credential file not found at {CREDENTIAL_FILE}. Server started - authentication will be required on first request.")
        return

    try:
        with open(CREDENTIAL_FILE, "r") as f:
            creds_data = json.load(f)
        
        if isinstance(creds_data, list) and creds_data:
            ACCOUNTS = creds_data
            account_cycler = itertools.cycle(ACCOUNTS)
            logging.info(f"Successfully loaded {len(ACCOUNTS)} accounts for polling.")
        elif isinstance(creds_data, dict):
             # Support single account format for backward compatibility
            ACCOUNTS = [creds_data]
            account_cycler = itertools.cycle(ACCOUNTS)
            logging.info("Successfully loaded 1 account.")
        else:
            logging.error("Credential file is not a valid JSON array of accounts or a single account object.")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse credentials file {CREDENTIAL_FILE}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading accounts: {e}")

# Load accounts when the module is first imported (i.e., when a worker process starts)
_load_accounts()


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    auth_code = None
    def do_GET(self):
        query_components = parse_qs(urlparse(self.path).query)
        code = query_components.get("code", [None])[0]
        if code:
            _OAuthCallbackHandler.auth_code = code
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>OAuth authentication successful!</h1><p>You can close this window. Please check the proxy server logs to verify that onboarding completed successfully. No need to restart the proxy.</p>")
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authentication failed.</h1><p>Please try again.</p>")


def authenticate_user(request: Request):
    """Authenticate the user with multiple methods."""
    # Check for API key in query parameters first (for Gemini client compatibility)
    api_key = request.query_params.get("key")
    if api_key and api_key == GEMINI_AUTH_PASSWORD:
        return "api_key_user"
    
    # Check for API key in x-goog-api-key header (Google SDK format)
    goog_api_key = request.headers.get("x-goog-api-key", "")
    if goog_api_key and goog_api_key == GEMINI_AUTH_PASSWORD:
        return "goog_api_key_user"
    
    # Check for API key in Authorization header (Bearer token format)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[7:]
        if bearer_token == GEMINI_AUTH_PASSWORD:
            return "bearer_user"
    
    # Check for HTTP Basic Authentication
    if auth_header.startswith("Basic "):
        try:
            encoded_credentials = auth_header[6:]
            decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8', "ignore")
            username, password = decoded_credentials.split(':', 1)
            if password == GEMINI_AUTH_PASSWORD:
                return username
        except Exception:
            pass
    
    # If none of the authentication methods work
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials. Use HTTP Basic Auth, Bearer token, 'key' query parameter, or 'x-goog-api-key' header.",
        headers={"WWW-Authenticate": "Basic"},
    )


def save_credentials(creds, project_id=None):
    """
    Saves the updated credentials for a specific account back to the file.
    This is critical for persisting refreshed tokens.
    """
    with file_lock:
        # Re-read the file to avoid race conditions with other processes
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                current_accounts = json.load(f)
                if not isinstance(current_accounts, list):
                    current_accounts = [current_accounts]
        except (FileNotFoundError, json.JSONDecodeError):
            current_accounts = []

        account_found = False
        for i, acc in enumerate(current_accounts):
            # Refresh token is the unique identifier for an account
            if acc.get("refresh_token") == creds.refresh_token:
                current_accounts[i]["token"] = creds.token
                if creds.expiry:
                    # Ensure expiry is in UTC and ISO format
                    expiry_utc = creds.expiry.astimezone(timezone.utc) if creds.expiry.tzinfo else creds.expiry.replace(tzinfo=timezone.utc)
                    current_accounts[i]["expiry"] = expiry_utc.isoformat()
                if project_id:
                    current_accounts[i]["project_id"] = project_id
                account_found = True
                break
        
        if not account_found:
            logging.warning("Could not find matching account to save refreshed credentials. This should not happen.")
            return

        try:
            with open(CREDENTIAL_FILE, "w") as f:
                # If there was only one account originally, save it as an object, otherwise as an array
                if len(current_accounts) == 1:
                    json.dump(current_accounts[0], f, indent=2)
                else:
                    json.dump(current_accounts, f, indent=2)
            logging.info(f"Successfully saved refreshed token for project {project_id or 'unknown'}.")
        except Exception as e:
            logging.error(f"Failed to write updated credentials to file: {e}")


def get_credentials(allow_oauth_flow=True):
    """
    Gets the next available account's credentials from the polling cycle.
    Handles token refresh and persists the new token.
    """
    global account_cycler
    if not account_cycler:
        # This fallback is for when the file was missing at startup, and the user logs in manually
        return _manual_oauth_flow() if allow_oauth_flow else None

    # Get the next account from the cycle
    selected_account = next(account_cycler)
    
    try:
        # The google-auth library expects specific keys, we ensure they exist.
        creds_info = selected_account.copy()
        if 'access_token' in creds_info and 'token' not in creds_info:
            creds_info['token'] = creds_info['access_token']
        if 'scope' in creds_info and 'scopes' not in creds_info:
            creds_info['scopes'] = creds_info['scope'].split()
        
        credentials = Credentials.from_authorized_user_info(creds_info, SCOPES)
        
        # Refresh credentials if needed
        if credentials.expired and credentials.refresh_token:
            try:
                logging.info(f"Token for project {selected_account.get('project_id')} expired. Refreshing...")
                credentials.refresh(GoogleAuthRequest())
                # Persist the newly refreshed token
                save_credentials(credentials, selected_account.get('project_id'))
            except Exception as e:
                logging.error(f"Failed to refresh token for project {selected_account.get('project_id')}: {e}")
                # Potentially remove this account from the cycle or mark as bad, but for now just log and continue
        
        return credentials

    except Exception as e:
        logging.error(f"Failed to create credentials from account data {selected_account.get('project_id')}: {e}")
        return None


def get_user_project_id(creds):
    """
    Gets the user's project ID. It now prioritizes the project ID from the
    account that was just selected by the polling mechanism.
    """
    # Priority 1: Environment variable always takes precedence
    env_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if env_project_id:
        logging.info(f"Using project ID from GOOGLE_CLOUD_PROJECT environment variable: {env_project_id}")
        return env_project_id

    # Priority 2: Find the project_id from the account matching the current credentials
    if creds and creds.refresh_token:
        with file_lock:
            # CORRECTED INDENTATION ON THE LINE BELOW
            for acc in ACCOUNTS:
                if acc.get("refresh_token") == creds.refresh_token:
                    if acc.get("project_id"):
                        logging.info(f"Using project_id for this request: {acc['project_id']}")
                        return acc["project_id"]
                    # If account is found but no project_id, break to use API discovery
                    break
    
    # Priority 3: Fallback to API discovery if project_id is missing in the file
    logging.warning("Could not find project_id in the selected account data, attempting API discovery.")
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
            "User-Agent": get_user_agent(),
        }
        probe_payload = {"metadata": get_client_metadata()}
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            data=json.dumps(probe_payload),
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        discovered_project_id = data.get("cloudaicompanionProject")
        if discovered_project_id:
            logging.info(f"Discovered project ID via API: {discovered_project_id}")
            # Save the discovered project_id back to the file for future use
            save_credentials(creds, discovered_project_id)
            logging.info(f"Using project_id for this request: {discovered_project_id}")
            return discovered_project_id
        else:
            raise ValueError("Could not find 'cloudaicompanionProject' in loadCodeAssist response.")
    except Exception as e:
        raise Exception(f"Failed to discover project ID via API: {e}")


def onboard_user(creds, project_id):
    """Ensures the user is onboarded, matching gemini-cli setupUser behavior."""
    global onboarding_complete_map
    if onboarding_complete_map.get(project_id):
        return

    # No need to re-check for expiration, get_credentials already handled it
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    
    load_assist_payload = {
        "cloudaicompanionProject": project_id,
        "metadata": get_client_metadata(project_id),
    }
    
    try:
        import requests
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            data=json.dumps(load_assist_payload),
            headers=headers,
        )
        resp.raise_for_status()
        load_data = resp.json()
        
        if load_data.get("currentTier"):
            onboarding_complete_map[project_id] = True
            return

        tier = None
        for allowed_tier in load_data.get("allowedTiers", []):
            if allowed_tier.get("isDefault"):
                tier = allowed_tier
                break
        
        if not tier:
            tier = {"id": "legacy-tier"}

        onboard_req_payload = {
            "tierId": tier.get("id"),
            "cloudaicompanionProject": project_id,
            "metadata": get_client_metadata(project_id),
        }
        
        onboard_resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
            data=json.dumps(onboard_req_payload),
            headers=headers,
        )
        onboard_resp.raise_for_status()
        # Assume onboarding is successful if the request doesn't fail
        onboarding_complete_map[project_id] = True

    except requests.exceptions.HTTPError as e:
        raise Exception(f"User onboarding failed for project {project_id}. Error: {e.response.text if hasattr(e, 'response') else str(e)}")
    except Exception as e:
        raise Exception(f"User onboarding failed for project {project_id} due to an unexpected error: {str(e)}")


def _manual_oauth_flow():
    """Initiates the manual OAuth flow if no credentials file is found."""
    client_config = {"installed": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token"}}
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri="http://localhost:8989")
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent", include_granted_scopes='true')
    
    print(f"\n{'='*80}\nAUTHENTICATION REQUIRED\n{'='*80}\nPlease open this URL in your browser to log in:\n{auth_url}\n{'='*80}\n")
    logging.info(f"Please open this URL in your browser to log in: {auth_url}")
    
    server = HTTPServer(("", 8989), _OAuthCallbackHandler)
    server.handle_request()
    
    auth_code = _OAuthCallbackHandler.auth_code
    if not auth_code:
        return None

    try:
        flow.fetch_token(code=auth_code)
        # Manually logged in, now save this as the first account in the file
        new_creds = flow.credentials
        
        # Discover project ID for the new account
        try:
            proj_id = get_user_project_id(new_creds)
        except Exception as e:
            proj_id = None
            logging.error(f"Could not discover project ID during initial login: {e}")

        # Save as a single object in the file
        creds_data = {
            "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "token": new_creds.token,
            "refresh_token": new_creds.refresh_token, "scopes": new_creds.scopes,
            "token_uri": "https://oauth2.googleapis.com/token",
            "expiry": new_creds.expiry.isoformat(),
            "project_id": proj_id
        }
        with open(CREDENTIAL_FILE, "w") as f:
            json.dump(creds_data, f, indent=2)
        
        logging.info("Authentication successful! Credentials saved.")
        _load_accounts() # Reload accounts now that the file exists
        return new_creds
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        return None
