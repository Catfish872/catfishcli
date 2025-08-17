import os
import json
import base64
import time
import logging
from datetime import datetime
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

# --- Global State ---
credentials = None
polling_credentials = []  # 用于存储从文件中加载的多个凭据
polling_index = 0         # 当前轮询到的凭据索引
user_project_id = None
onboarding_complete = False
credentials_from_env = False  # Track if credentials came from environment variable

security = HTTPBasic()

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
    global credentials_from_env
    
    # Don't save credentials to file if they came from environment variable,
    # but still save project_id if provided and no file exists or file lacks project_id
    if credentials_from_env:
        if project_id and os.path.exists(CREDENTIAL_FILE):
            try:
                with open(CREDENTIAL_FILE, "r") as f:
                    existing_data = json.load(f)
                # Only update project_id if it's missing from the file
                if "project_id" not in existing_data:
                    existing_data["project_id"] = project_id
                    with open(CREDENTIAL_FILE, "w") as f:
                        json.dump(existing_data, f, indent=2)
                    logging.info(f"Added project_id {project_id} to existing credential file")
            except Exception as e:
                logging.warning(f"Could not update project_id in credential file: {e}")
        return
    
    creds_data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "scopes": creds.scopes if creds.scopes else SCOPES,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    
    if creds.expiry:
        if creds.expiry.tzinfo is None:
            from datetime import timezone
            expiry_utc = creds.expiry.replace(tzinfo=timezone.utc)
        else:
            expiry_utc = creds.expiry
        # Keep the existing ISO format for backward compatibility, but ensure it's properly handled during loading
        creds_data["expiry"] = expiry_utc.isoformat()
    
    if project_id:
        creds_data["project_id"] = project_id
    elif os.path.exists(CREDENTIAL_FILE):
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                existing_data = json.load(f)
                if "project_id" in existing_data:
                    creds_data["project_id"] = existing_data["project_id"]
        except Exception:
            pass
    
    
    with open(CREDENTIAL_FILE, "w") as f:
        json.dump(creds_data, f, indent=2)
    

def get_credentials(allow_oauth_flow=True):
    """Loads credentials matching gemini-cli OAuth2 flow."""
    # 引用全局变量
    global credentials, credentials_from_env, user_project_id, polling_credentials, polling_index
    
    # --- BUG FIX ---
    # The original caching logic below is REMOVED because it breaks polling.
    # In a polling scenario, we MUST re-evaluate which credential to use on every call.
    #
    # DELETED CODE:
    # if credentials and credentials.token and not credentials.expired:
    #     return credentials

    # Check for credentials file (CREDENTIAL_FILE is derived from GOOGLE_APPLICATION_CREDENTIALS)
    if os.path.exists(CREDENTIAL_FILE):
        try:
            # If our polling list is empty, load it once from the file.
            if not polling_credentials:
                with open(CREDENTIAL_FILE, "r") as f:
                    loaded_json = json.load(f)
                
                if isinstance(loaded_json, list):
                    if not loaded_json:
                        raise ValueError(f"The credential file '{CREDENTIAL_FILE}' is an empty list.")
                    polling_credentials = loaded_json
                    logging.info(f"Successfully loaded {len(polling_credentials)} polling credentials from file.")
                elif isinstance(loaded_json, dict):
                    polling_credentials = [loaded_json]
                    logging.info(f"Detected a single credential object in file. Loaded in compatibility mode.")
                else:
                    raise ValueError(f"The format of '{CREDENTIAL_FILE}' is not recognized. It must be a JSON object or a JSON array.")

            # --- Core Polling Logic ---
            selected_cred_info = None
            if polling_credentials:
                import threading
                with threading.Lock():
                    selected_cred_info = polling_credentials[polling_index]
                    polling_index = (polling_index + 1) % len(polling_credentials)
            
            if not selected_cred_info:
                raise ValueError("Polling credentials list is empty or invalid.")

            raw_creds_data = selected_cred_info
            
            if "refresh_token" in raw_creds_data and raw_creds_data["refresh_token"]:
                current_cred_index = (polling_index - 1 + len(polling_credentials)) % len(polling_credentials)
                logging.info(f"Using credential at index {current_cred_index}.")
                
                try:
                    creds_data = raw_creds_data.copy()
                    
                    if "access_token" in creds_data and "token" not in creds_data:
                        creds_data["token"] = creds_data["access_token"]
                    if "scope" in creds_data and "scopes" not in creds_data:
                        creds_data["scopes"] = creds_data["scope"].split()
                    if "expiry" in creds_data:
                        expiry_str = creds_data["expiry"]
                        if isinstance(expiry_str, str) and ("+00:00" in expiry_str or "Z" in expiry_str):
                            try:
                                from datetime import datetime
                                if "+00:00" in expiry_str:
                                    parsed_expiry = datetime.fromisoformat(expiry_str)
                                elif expiry_str.endswith("Z"):
                                    parsed_expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                                else:
                                    parsed_expiry = datetime.fromisoformat(expiry_str)
                                
                                import time
                                timestamp = parsed_expiry.timestamp()
                                creds_data["expiry"] = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
                            except Exception:
                                del creds_data["expiry"]
                    
                    # Create a NEW credentials object for this specific request
                    request_credentials = Credentials.from_authorized_user_info(creds_data, SCOPES)
                    credentials_from_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

                    if request_credentials.expired and request_credentials.refresh_token:
                        try:
                            logging.info(f"Credential at index {current_cred_index} is expired, attempting refresh...")
                            request_credentials.refresh(GoogleAuthRequest())
                            logging.info(f"Credential at index {current_cred_index} refreshed successfully.")
                        except Exception as refresh_error:
                            logging.warning(f"Failed to refresh credentials: {refresh_error}")
                    
                    # Return the newly created/refreshed credential object for this request
                    return request_credentials
                    
                except Exception as parsing_error:
                    logging.warning(f"Failed to parse selected credential, attempting minimal creation. Error: {parsing_error}")
                    try:
                        minimal_creds_data = {
                            "client_id": raw_creds_data.get("client_id", CLIENT_ID),
                            "client_secret": raw_creds_data.get("client_secret", CLIENT_SECRET),
                            "refresh_token": raw_creds_data["refresh_token"],
                            "token_uri": "https://oauth2.googleapis.com/token",
                        }
                        
                        request_credentials = Credentials.from_authorized_user_info(minimal_creds_data, SCOPES)
                        credentials_from_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
                        
                        logging.info("Refreshing minimal credentials...")
                        request_credentials.refresh(GoogleAuthRequest())
                        logging.info("Minimal credentials refreshed successfully.")
                        return request_credentials
                    except Exception as minimal_error:
                        logging.error(f"Failed to create and refresh minimal credentials: {minimal_error}")
            else:
                logging.warning("No refresh token found in the selected credential from the polling file.")
                
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error processing credential file '{CREDENTIAL_FILE}': {e}")

    # Fallback to the interactive OAuth flow if no valid credentials could be loaded from file
    if not allow_oauth_flow:
        logging.info("OAuth flow not allowed - returning None.")
        return None

    client_config = {
        "installed": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri="http://localhost:8989"
    )
    
    flow.oauth2session.scope = SCOPES
    
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes='true'
    )
    print(f"\n{'='*80}")
    print(f"AUTHENTICATION REQUIRED")
    print(f"{'='*80}")
    print(f"Please open this URL in your browser to log in:")
    print(f"{auth_url}")
    print(f"{'='*80}\n")
    logging.info(f"Please open this URL in your browser to log in: {auth_url}")
    
    server = HTTPServer(("", 8989), _OAuthCallbackHandler)
    server.handle_request()
    
    auth_code = _OAuthCallbackHandler.auth_code
    if not auth_code:
        return None

    import oauthlib.oauth2.rfc6749.parameters
    original_validate = oauthlib.oauth2.rfc6749.parameters.validate_token_parameters
    
    def patched_validate(params):
        try:
            return original_validate(params)
        except Warning:
            pass
    
    oauthlib.oauth2.rfc6749.parameters.validate_token_parameters = patched_validate
    
    try:
        credentials = flow.credentials
        credentials_from_env = False
        save_credentials(credentials)
        logging.info("Authentication successful! Credentials saved.")
        return credentials
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        return None
    finally:
        oauthlib.oauth2.rfc6749.parameters.validate_token_parameters = original_validate

def onboard_user(creds, project_id):
    """Ensures the user is onboarded, matching gemini-cli setupUser behavior."""
    global onboarding_complete
    if onboarding_complete:
        return

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            save_credentials(creds)
        except Exception as e:
            raise Exception(f"Failed to refresh credentials during onboarding: {str(e)}")
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
        
        tier = None
        if load_data.get("currentTier"):
            tier = load_data["currentTier"]
        else:
            for allowed_tier in load_data.get("allowedTiers", []):
                if allowed_tier.get("isDefault"):
                    tier = allowed_tier
                    break
            
            if not tier:
                tier = {
                    "name": "",
                    "description": "",
                    "id": "legacy-tier",
                    "userDefinedCloudaicompanionProject": True,
                }

        if tier.get("userDefinedCloudaicompanionProject") and not project_id:
            raise ValueError("This account requires setting the GOOGLE_CLOUD_PROJECT env var.")

        if load_data.get("currentTier"):
            onboarding_complete = True
            return

        onboard_req_payload = {
            "tierId": tier.get("id"),
            "cloudaicompanionProject": project_id,
            "metadata": get_client_metadata(project_id),
        }

        while True:
            onboard_resp = requests.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                data=json.dumps(onboard_req_payload),
                headers=headers,
            )
            onboard_resp.raise_for_status()
            lro_data = onboard_resp.json()

            if lro_data.get("done"):
                onboarding_complete = True
                break
            
            time.sleep(5)

    except requests.exceptions.HTTPError as e:
        raise Exception(f"User onboarding failed. Please check your Google Cloud project permissions and try again. Error: {e.response.text if hasattr(e, 'response') else str(e)}")
    except Exception as e:
        raise Exception(f"User onboarding failed due to an unexpected error: {str(e)}")

def get_user_project_id(creds):
    """Gets the user's project ID matching gemini-cli setupUser logic."""
    global user_project_id
    
    # Priority 1: Check environment variable first (always check, even if user_project_id is set)
    env_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if env_project_id:
        logging.info(f"Using project ID from GOOGLE_CLOUD_PROJECT environment variable: {env_project_id}")
        user_project_id = env_project_id
        save_credentials(creds, user_project_id)
        return user_project_id
    
    # If we already have a cached project_id and no env var override, use it
    if user_project_id:
        logging.info(f"Using cached project ID: {user_project_id}")
        return user_project_id

    # Priority 2: Check cached project ID in credential file
    if os.path.exists(CREDENTIAL_FILE):
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                creds_data = json.load(f)
                cached_project_id = creds_data.get("project_id")
                if cached_project_id:
                    logging.info(f"Using cached project ID from credential file: {cached_project_id}")
                    user_project_id = cached_project_id
                    return user_project_id
        except Exception as e:
            logging.warning(f"Could not read project_id from credential file: {e}")

    # Priority 3: Make API call to discover project ID
    # Ensure we have valid credentials for the API call
    if creds.expired and creds.refresh_token:
        try:
            logging.info("Refreshing credentials before project ID discovery...")
            creds.refresh(GoogleAuthRequest())
            save_credentials(creds)
            logging.info("Credentials refreshed successfully for project ID discovery")
        except Exception as e:
            logging.error(f"Failed to refresh credentials while getting project ID: {e}")
            # Continue with existing credentials - they might still work
    
    if not creds.token:
        raise Exception("No valid access token available for project ID discovery")
    
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    
    probe_payload = {
        "metadata": get_client_metadata(),
    }

    try:
        import requests
        logging.info("Attempting to discover project ID via API call...")
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            data=json.dumps(probe_payload),
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        discovered_project_id = data.get("cloudaicompanionProject")
        if not discovered_project_id:
            raise ValueError("Could not find 'cloudaicompanionProject' in loadCodeAssist response.")

        logging.info(f"Discovered project ID via API: {discovered_project_id}")
        user_project_id = discovered_project_id
        save_credentials(creds, user_project_id)
        
        return user_project_id
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error during project ID discovery: {e}")
        if hasattr(e, 'response') and e.response:
            logging.error(f"Response status: {e.response.status_code}, body: {e.response.text}")
        raise Exception(f"Failed to discover project ID via API: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during project ID discovery: {e}")
        raise Exception(f"Failed to discover project ID: {e}")
