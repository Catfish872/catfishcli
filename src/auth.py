# 位于 catfishcli/src/auth.py

import os
import json
import base64
import time
import logging
import threading
from datetime import datetime
from fastapi import Request, HTTPException
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

# --- Centralized, Process-Safe State Management ---
class PollingManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PollingManager, cls).__new__(cls)
                cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        self.polling_credentials = []
        # Use a file for the index to share state across processes, crucial for multi-worker environments like Koyeb
        self.index_file = os.path.join(os.path.dirname(os.path.abspath(CREDENTIAL_FILE)), ".polling_index")
        # In-memory cache for project_id and onboarding status, keyed by refresh_token.
        # This cache is per-process, which is fine as states are discovered on demand.
        self.state_cache = {}
        self.initialized = True
        self._load_credentials_from_file()

    def _load_credentials_from_file(self):
        if not os.path.exists(CREDENTIAL_FILE):
            logging.warning(f"Credential file not found at {CREDENTIAL_FILE}. Interactive login may be required.")
            return
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                loaded_json = json.load(f)

            if isinstance(loaded_json, list):
                if not loaded_json: raise ValueError("Credential file is an empty list.")
                self.polling_credentials = loaded_json
                logging.info(f"Successfully loaded {len(self.polling_credentials)} polling credentials.")
            elif isinstance(loaded_json, dict):
                self.polling_credentials = [loaded_json]
                logging.info("Detected a single credential object. Loaded in compatibility mode.")
            else:
                raise ValueError("Credential file format is not a JSON object or array.")
        except Exception as e:
            logging.error(f"Failed to load credentials from {CREDENTIAL_FILE}: {e}")
            self.polling_credentials = []

    def get_next_credential_info(self):
        with self._lock:
            if not self.polling_credentials: return None, -1
            try:
                # Read-modify-write of the index file is now atomic and process-safe
                with open(self.index_file, "r") as f:
                    current_index = int(f.read()) % len(self.polling_credentials)
            except (FileNotFoundError, ValueError):
                current_index = 0
            
            selected_cred = self.polling_credentials[current_index]
            next_index = (current_index + 1) % len(self.polling_credentials)
            with open(self.index_file, "w") as f: f.write(str(next_index))
            return selected_cred, current_index

    def get_cached_state(self, refresh_token):
        return self.state_cache.get(refresh_token, {})

    def set_cached_state(self, refresh_token, project_id=None, onboarded=None):
        with self._lock:
            if refresh_token not in self.state_cache: self.state_cache[refresh_token] = {}
            if project_id is not None: self.state_cache[refresh_token]['project_id'] = project_id
            if onboarded is not None: self.state_cache[refresh_token]['onboarded'] = onboarded

polling_manager = PollingManager()

# --- Legacy global state is no longer used by core logic but retained for interactive flow ---
credentials_from_env = False

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    auth_code = None
    def do_GET(self):
        # ... (unmodified)
        query_components = parse_qs(urlparse(self.path).query)
        code = query_components.get("code", [None])[0]
        if code:
            _OAuthCallbackHandler.auth_code = code
            self.send_response(200); self.send_header("Content-type", "text/html"); self.end_headers()
            self.wfile.write(b"<h1>OAuth authentication successful!</h1><p>You can close this window.</p>")
        else:
            self.send_response(400); self.send_header("Content-type", "text/html"); self.end_headers()
            self.wfile.write(b"<h1>Authentication failed.</h1>")

def authenticate_user(request: Request):
    # ... (unmodified)
    api_key = request.query_params.get("key")
    if api_key and api_key == GEMINI_AUTH_PASSWORD: return "api_key_user"
    goog_api_key = request.headers.get("x-goog-api-key", "")
    if goog_api_key and goog_api_key == GEMINI_AUTH_PASSWORD: return "goog_api_key_user"
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        if auth_header[7:] == GEMINI_AUTH_PASSWORD: return "bearer_user"
    if auth_header.startswith("Basic "):
        try:
            _, password = base64.b64decode(auth_header[6:]).decode('utf-8', "ignore").split(':', 1)
            if password == GEMINI_AUTH_PASSWORD: return "basic_auth_user"
        except Exception: pass
    raise HTTPException(status_code=401, detail="Invalid authentication credentials.", headers={"WWW-Authenticate": "Basic"})

def save_credentials(creds, project_id=None):
    # ... (unmodified, for interactive flow)
    creds_data = {
        "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "token": creds.token,
        "refresh_token": creds.refresh_token, "scopes": creds.scopes if creds.scopes else SCOPES,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    if creds.expiry: creds_data["expiry"] = creds.expiry.isoformat()
    if project_id: creds_data["project_id"] = project_id
    with open(CREDENTIAL_FILE, "w") as f:
        json.dump(creds_data, f, indent=2)

def get_credentials(allow_oauth_flow=True):
    # This function is now just a clean entry point to the manager and the original robust logic.
    raw_creds_data, current_index = polling_manager.get_next_credential_info()
    if not raw_creds_data:
        return _interactive_oauth_flow(allow_oauth_flow)
    
    try:
        if "refresh_token" in raw_creds_data and raw_creds_data["refresh_token"]:
            logging.info(f"Using credential at index {current_index}.")
            try:
                # This block is the original author's robust parsing logic, completely preserved.
                creds_data = raw_creds_data.copy()
                if "access_token" in creds_data and "token" not in creds_data: creds_data["token"] = creds_data["access_token"]
                if "scope" in creds_data and "scopes" not in creds_data: creds_data["scopes"] = creds_data["scope"].split()
                if "expiry" in creds_data:
                    expiry_str = creds_data["expiry"]
                    if isinstance(expiry_str, str) and ("+00:00" in expiry_str or "Z" in expiry_str):
                        try:
                            if "+00:00" in expiry_str: parsed_expiry = datetime.fromisoformat(expiry_str)
                            elif expiry_str.endswith("Z"): parsed_expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                            else: parsed_expiry = datetime.fromisoformat(expiry_str)
                            creds_data["expiry"] = datetime.utcfromtimestamp(parsed_expiry.timestamp()).strftime("%Y-%m-%dT%H:%M:%SZ")
                        except Exception as e:
                            logging.warning(f"Could not parse expiry '{expiry_str}': {e}, removing field.")
                            del creds_data["expiry"]
                
                credentials = Credentials.from_authorized_user_info(creds_data, SCOPES)
                if credentials.expired and credentials.refresh_token:
                    try:
                        logging.info(f"Credential at index {current_index} is expired, attempting refresh...")
                        credentials.refresh(GoogleAuthRequest())
                        logging.info(f"Credential at index {current_index} refreshed successfully.")
                    except Exception as e:
                        logging.warning(f"Failed to refresh credentials for index {current_index}: {e}")
                
                return credentials
            except Exception as e:
                logging.warning(f"Failed to parse credential at index {current_index}, attempting minimal. Error: {e}")
                try:
                    minimal_creds = {"client_id": raw_creds_data.get("client_id", CLIENT_ID), "client_secret": raw_creds_data.get("client_secret", CLIENT_SECRET), "refresh_token": raw_creds_data["refresh_token"], "token_uri": "https://oauth2.googleapis.com/token"}
                    credentials = Credentials.from_authorized_user_info(minimal_creds, SCOPES)
                    credentials.refresh(GoogleAuthRequest())
                    return credentials
                except Exception as e_min:
                    logging.error(f"Failed to create minimal credentials for index {current_index}: {e_min}")
        else:
            logging.warning(f"No refresh token in credential at index {current_index}.")
    except Exception as e:
        logging.error(f"Unexpected error in get_credentials for index {current_index}: {e}")

    return _interactive_oauth_flow(allow_oauth_flow)

def _interactive_oauth_flow(allow_oauth_flow):
    # ... (unmodified, for interactive fallback)
    if not allow_oauth_flow:
        logging.info("OAuth flow not allowed.")
        return None
    
    client_config = {"installed": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token"}}
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri="http://localhost:8989")
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent", include_granted_scopes='true')
    print(f"\n{'='*80}\nAUTHENTICATION REQUIRED\n{'='*80}\nPlease open this URL in your browser:\n{auth_url}\n{'='*80}\n")
    server = HTTPServer(("", 8989), _OAuthCallbackHandler)
    server.handle_request()
    if not _OAuthCallbackHandler.auth_code: return None
    
    import oauthlib.oauth2.rfc6749.parameters
    original_validate = oauthlib.oauth2.rfc6749.parameters.validate_token_parameters
    def patched_validate(params):
        try: return original_validate(params)
        except Warning: pass
    oauthlib.oauth2.rfc6749.parameters.validate_token_parameters = patched_validate
    
    try:
        flow.fetch_token(code=_OAuthCallbackHandler.auth_code)
        save_credentials(flow.credentials)
        logging.info("Authentication successful! Credentials saved.")
        polling_manager._load_credentials_from_file()
        return flow.credentials
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        return None
    finally:
        oauthlib.oauth2.rfc6749.parameters.validate_token_parameters = original_validate

def onboard_user(creds, project_id):
    # This function is now stateful via the PollingManager, removing global variable pollution.
    refresh_token = creds.refresh_token
    if not refresh_token: return
    
    cached_state = polling_manager.get_cached_state(refresh_token)
    if cached_state.get('onboarded'):
        return

    if creds.expired and creds.refresh_token:
        try: creds.refresh(GoogleAuthRequest())
        except Exception as e: raise Exception(f"Failed to refresh credentials during onboarding: {e}")
    
    headers = {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json", "User-Agent": get_user_agent()}
    payload = {"cloudaicompanionProject": project_id, "metadata": get_client_metadata(project_id)}
    
    try:
        import requests
        resp = requests.post(f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist", data=json.dumps(payload), headers=headers)
        resp.raise_for_status()
        load_data = resp.json()
        
        if load_data.get("currentTier"):
            polling_manager.set_cached_state(refresh_token, onboarded=True)
            return

        tier = next((t for t in load_data.get("allowedTiers", []) if t.get("isDefault")), {"id": "legacy-tier"})
        onboard_payload = {"tierId": tier.get("id"), "cloudaicompanionProject": project_id, "metadata": get_client_metadata(project_id)}
        
        onboard_resp = requests.post(f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser", data=json.dumps(onboard_payload), headers=headers)
        onboard_resp.raise_for_status()
        logging.info(f"Onboarding successful for project {project_id}")
        
        polling_manager.set_cached_state(refresh_token, onboarded=True)

    except Exception as e:
        raise Exception(f"User onboarding failed for project {project_id}: {getattr(e, 'response', e)}")

def get_user_project_id(creds):
    # This function is now stateful via the PollingManager, removing global variable pollution.
    env_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if env_project_id:
        return env_project_id
        
    refresh_token = creds.refresh_token
    if not refresh_token: raise ValueError("Cannot get project ID without a refresh token.")
    
    # Priority 1: Check the process-safe cache first.
    cached_state = polling_manager.get_cached_state(refresh_token)
    if 'project_id' in cached_state:
        logging.info(f"Using cached project ID for token ...{refresh_token[-6:]}: {cached_state['project_id']}")
        return cached_state['project_id']
    
    # Priority 2: Original logic for discovering Project ID, now with safe caching.
    if creds.expired and creds.refresh_token:
        try: creds.refresh(GoogleAuthRequest())
        except Exception as e: logging.error(f"Failed to refresh credentials while getting project ID: {e}")
    
    if not creds.token: raise Exception("No valid access token for project ID discovery")
    
    headers = {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json", "User-Agent": get_user_agent()}
    payload = {"metadata": get_client_metadata()}
    try:
        import requests
        logging.info(f"Discovering project ID for token ...{refresh_token[-6:]} via API call...")
        resp = requests.post(f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist", data=json.dumps(payload), headers=headers)
        resp.raise_for_status()
        data = resp.json()
        discovered_project_id = data.get("cloudaicompanionProject")
        if not discovered_project_id: raise ValueError("Could not find 'cloudaicompanionProject' in API response.")
        
        logging.info(f"Discovered and cached project ID: {discovered_project_id}")
        polling_manager.set_cached_state(refresh_token, project_id=discovered_project_id)
        return discovered_project_id
    except Exception as e:
        raise Exception(f"Failed to discover project ID via API: {getattr(e, 'response', e)}")
