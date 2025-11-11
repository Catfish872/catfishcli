"""
Google API Client - Handles all communication with Google's Gemini API.
This module is used by both OpenAI compatibility layer and native Gemini endpoints.
"""
import json
import logging
import requests
from fastapi import Response
from fastapi.responses import StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from .project_poller import get_next_project_id
from .config import GEMINI_RETRY_COUNT

from .auth import get_credentials, save_credentials, get_user_project_id, onboard_user
from .utils import get_user_agent
from .config import (
    CODE_ASSIST_ENDPOINT,
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name,
    is_search_model,
    get_thinking_budget,
    should_include_thoughts
)
import asyncio
from datetime import date
import threading

# 使用全局字典在内存中存储统计信息
usage_stats = {
    "last_reset_date": date.today(),
    "total_success": 0,
    "total_fail": 0,
    "accounts": {}  # key: project_id, value: {"success": 0, "fail": 0}
}
# 线程锁，用于保证并发请求时数据更新的原子性
stats_lock = threading.Lock()


def _check_and_reset_stats():
    """检查日期，如果到了新的一天，则重置所有计数器。必须在锁内调用。"""
    global usage_stats
    today = date.today()
    if usage_stats["last_reset_date"] != today:
        logging.info(f"New day detected. Resetting usage statistics from {usage_stats['last_reset_date']} to {today}.")
        usage_stats = {
            "last_reset_date": today,
            "total_success": 0,
            "total_fail": 0,
            "accounts": {}
        }


def record_usage(project_id: str, success: bool):
    """记录一次API调用（成功或失败）。这是一个线程安全的函数。"""
    if not project_id:  # 防止 project_id 为 None 时出错
        project_id = "unknown_project"
    with stats_lock:
        _check_and_reset_stats()
        usage_stats["accounts"].setdefault(project_id, {"success": 0, "fail": 0})

        if success:
            usage_stats["total_success"] += 1
            usage_stats["accounts"][project_id]["success"] += 1
        else:
            usage_stats["total_fail"] += 1
            usage_stats["accounts"][project_id]["fail"] += 1


def get_formatted_stats() -> str:
    """获取格式化为文本表格的统计信息字符串。这是一个线程安全的函数。"""
    with stats_lock:
        _check_and_reset_stats()
        stats_copy = usage_stats.copy()
        accounts_copy = stats_copy["accounts"].copy()

    header = (
        f"--- Daily Usage Statistics (Last Reset: {stats_copy['last_reset_date']}) ---\n"
        f"Overall: {stats_copy['total_success']} Success, {stats_copy['total_fail']} Fail\n"
        f"{'-' * 60}\n"
        f"| {'Project ID'.ljust(35)} | {'Success'.rjust(7)} | {'Fail'.rjust(7)} |\n"
        f"|{'-' * 37}|{'-' * 9}|{'-' * 9}|\n"
    )

    body = ""
    if not accounts_copy:
        body = "| No account usage recorded today.                                  |\n"
    else:
        for project_id, counts in sorted(accounts_copy.items()):
            body += f"| {str(project_id).ljust(35)} | {str(counts['success']).rjust(7)} | {str(counts['fail']).rjust(7)} |\n"

    footer = f"{'-' * 60}"
    return header + body + footer


def send_gemini_request(payload: dict, is_streaming: bool = False) -> Response:
    """
    Send a request to Google's Gemini API with retry and project polling logic.

    Args:
        payload: The request payload in Gemini format
        is_streaming: Whether this is a streaming request

    Returns:
        FastAPI Response object
    """
    # ===================== MODIFY START =====================
    last_error_response = None
    last_used_proj_id = None  # 用于在最终失败时记录

    # Total attempts = 1 (initial) + GEMINI_RETRY_COUNT
    for attempt in range(GEMINI_RETRY_COUNT + 1):
        # 功能一：每次重试都获取一个新账号凭据
        creds = get_credentials()
        if not creds:
            logging.error(f"Attempt {attempt + 1}: Failed to get credentials.")
            if attempt < GEMINI_RETRY_COUNT:
                continue
            else:
                last_error_response = Response(
                    content="Authentication failed. Could not retrieve any valid credentials.",
                    status_code=500
                )
                break

        # Refresh credentials if needed
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(GoogleAuthRequest())
                save_credentials(creds)
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}: Token refresh failed for an account. Trying next. Error: {e}")
                if attempt < GEMINI_RETRY_COUNT:
                    continue
                else:
                    last_error_response = Response(
                        content="Token refresh failed. Please restart the proxy to re-authenticate.",
                        status_code=500
                    )
                    break
        elif not creds.token:
            logging.warning(f"Attempt {attempt + 1}: Account has no access token. Trying next one.")
            if attempt < GEMINI_RETRY_COUNT:
                continue
            else:
                last_error_response = Response(
                    content="No access token. Please restart the proxy to re-authenticate.",
                    status_code=500
                )
                break

        # 使用与凭据绑定的项目ID，而不是轮询独立的列表
        proj_id = get_user_project_id(creds)
        last_used_proj_id = proj_id

        logging.info(f"Attempt {attempt + 1}/{GEMINI_RETRY_COUNT + 1}: Using account for project: {proj_id}")
        if not proj_id:
            logging.error("Failed to get a valid user project ID for this attempt.")
            if attempt < GEMINI_RETRY_COUNT:
                continue  # 尝试下一个账号
            else:
                last_error_response = Response(content="Could not determine project ID for any account.",
                                               status_code=500)
                break

        onboard_user(creds, proj_id)

        # Build the final payload with the current project info
        final_payload = {
            "model": payload.get("model"),
            "project": proj_id,
            "request": payload.get("request", {})
        }
        final_post_data = json.dumps(final_payload)

        # Determine the action and URL
        action = "streamGenerateContent" if is_streaming else "generateContent"
        target_url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}"
        if is_streaming:
            target_url += "?alt=sse"

        # Build request headers
        request_headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
            "User-Agent": get_user_agent(),
        }

        try:
            if is_streaming:
                resp = requests.post(target_url, data=final_post_data, headers=request_headers, stream=True)
            else:
                resp = requests.post(target_url, data=final_post_data, headers=request_headers)

            # Check for retry conditions
            is_429 = resp.status_code == 429
            is_empty_reply = False

            # Check for empty reply only on non-streaming, successful requests
            if not is_streaming and resp.status_code == 200:
                try:
                    # Peek into the response content to check if it's effectively empty
                    resp_text = resp.text
                    if resp_text.startswith('data: '):
                        resp_text = resp_text[len('data: '):]

                    if not resp_text.strip():
                        is_empty_reply = True
                    else:
                        api_response = json.loads(resp_text)
                        gemini_response = api_response.get("response", {})
                        candidates = gemini_response.get("candidates", [])

                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            has_main_text = any("text" in p and not p.get("thought") for p in parts)
                            has_tool_call = any("functionCall" in p for p in parts)

                            if not has_main_text and not has_tool_call:
                                is_empty_reply = True
                        else:
                            is_empty_reply = True

                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

            if is_429 or is_empty_reply:
                reason = "status 429" if is_429 else "empty reply"
                logging.warning(f"Attempt {attempt + 1} failed due to {reason}. Retrying...")
                last_error_response = resp
                if attempt < GEMINI_RETRY_COUNT:
                    continue
                else:
                    break

                    # If successful, handle and return immediately
            # 功能二：记录成功调用
            record_usage(proj_id, success=True)
            if is_streaming:
                return _handle_streaming_response(resp)
            else:
                return _handle_non_streaming_response(resp)

        except requests.exceptions.RequestException as e:
            logging.error(f"Request to Google API failed on attempt {attempt + 1}: {str(e)}")
            error_content = json.dumps(
                {"error": {"message": f"Request failed: {str(e)}", "type": "api_connection_error"}})
            last_error_response = Response(content=error_content, status_code=502, media_type="application/json")
            if attempt < GEMINI_RETRY_COUNT:
                continue
            else:
                break

    # If the loop finishes without a successful return, all retries have failed.
    logging.error("All retry attempts failed.")
    # 功能二：记录失败调用
    record_usage(last_used_proj_id, success=False)

    if last_error_response:
        # Return the last captured error response
        return _handle_non_streaming_response(last_error_response)
    else:
        # Fallback error if no response was ever received
        return Response(
            content=json.dumps({"error": {"message": "All retry attempts failed to connect to the upstream server."}}),
            status_code=503,
            media_type="application/json"
        )


def _handle_streaming_response(resp) -> StreamingResponse:
    """Handle streaming response from Google API."""
    
    # Check for HTTP errors before starting to stream
    if resp.status_code != 200:
        logging.error(f"Google API returned status {resp.status_code}: {resp.text}")
        error_message = f"Google API error: {resp.status_code}"
        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", error_message)
        except:
            pass
        
        # Return error as a streaming response
        async def error_generator():
            error_response = {
                "error": {
                    "message": error_message,
                    "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                    "code": resp.status_code
                }
            }
            yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8')
        
        response_headers = {
            "Content-Type": "text/event-stream",
            "Content-Disposition": "attachment",
            "Vary": "Origin, X-Origin, Referer",
            "X-XSS-Protection": "0",
            "X-Frame-Options": "SAMEORIGIN",
            "X-Content-Type-Options": "nosniff",
            "Server": "ESF"
        }
        
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            headers=response_headers,
            status_code=resp.status_code
        )
    
    async def stream_generator():
        try:
            with resp:
                for chunk in resp.iter_lines():
                    if chunk:
                        if not isinstance(chunk, str):
                            chunk = chunk.decode('utf-8', "ignore")
                            
                        if chunk.startswith('data: '):
                            chunk = chunk[len('data: '):]
                            
                            try:
                                obj = json.loads(chunk)
                                
                                if "response" in obj:
                                    response_chunk = obj["response"]
                                    response_json = json.dumps(response_chunk, separators=(',', ':'))
                                    response_line = f"data: {response_json}\n\n"
                                    yield response_line.encode('utf-8', "ignore")
                                    await asyncio.sleep(0)
                                else:
                                    obj_json = json.dumps(obj, separators=(',', ':'))
                                    yield f"data: {obj_json}\n\n".encode('utf-8', "ignore")
                            except json.JSONDecodeError:
                                continue
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Streaming request failed: {str(e)}")
            error_response = {
                "error": {
                    "message": f"Upstream request failed: {str(e)}",
                    "type": "api_error",
                    "code": 502
                }
            }
            yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8', "ignore")
        except Exception as e:
            logging.error(f"Unexpected error during streaming: {str(e)}")
            error_response = {
                "error": {
                    "message": f"An unexpected error occurred: {str(e)}",
                    "type": "api_error",
                    "code": 500
                }
            }
            yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8', "ignore")

    response_headers = {
        "Content-Type": "text/event-stream",
        "Content-Disposition": "attachment",
        "Vary": "Origin, X-Origin, Referer",
        "X-XSS-Protection": "0",
        "X-Frame-Options": "SAMEORIGIN",
        "X-Content-Type-Options": "nosniff",
        "Server": "ESF"
    }
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers=response_headers
    )


def _handle_non_streaming_response(resp) -> Response:
    """Handle non-streaming response from Google API."""
    if resp.status_code == 200:
        try:
            google_api_response = resp.text
            if google_api_response.startswith('data: '):
                google_api_response = google_api_response[len('data: '):]
            google_api_response = json.loads(google_api_response)
            standard_gemini_response = google_api_response.get("response")
            return Response(
                content=json.dumps(standard_gemini_response),
                status_code=200,
                media_type="application/json; charset=utf-8"
            )
        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Failed to parse Google API response: {str(e)}")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("Content-Type")
            )
    else:
        # Log the error details
        logging.error(f"Google API returned status {resp.status_code}: {resp.text}")
        
        # Try to parse error response and provide meaningful error message
        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", f"API error: {resp.status_code}")
                error_response = {
                    "error": {
                        "message": error_message,
                        "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                        "code": resp.status_code
                    }
                }
                return Response(
                    content=json.dumps(error_response),
                    status_code=resp.status_code,
                    media_type="application/json"
                )
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback to original response if we can't parse the error
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("Content-Type")
        )


def build_gemini_payload_from_openai(openai_payload: dict) -> dict:
    """
    Build a Gemini API payload from an OpenAI-transformed request.
    This is used when OpenAI requests are converted to Gemini format.
    """
    # Extract model from the payload
    model = openai_payload.get("model")
    
    # Get safety settings or use defaults
    safety_settings = openai_payload.get("safetySettings", DEFAULT_SAFETY_SETTINGS)
    
    # Build the request portion
    request_data = {
        "contents": openai_payload.get("contents"),
        "systemInstruction": openai_payload.get("systemInstruction"),
        "cachedContent": openai_payload.get("cachedContent"),
        "tools": openai_payload.get("tools"),
        "toolConfig": openai_payload.get("toolConfig"),
        "safetySettings": safety_settings,
        "generationConfig": openai_payload.get("generationConfig", {}),
    }
    
    # Remove any keys with None values
    request_data = {k: v for k, v in request_data.items() if v is not None}
    
    return {
        "model": model,
        "request": request_data
    }


def build_gemini_payload_from_native(native_request: dict, model_from_path: str) -> dict:
    """
    Build a Gemini API payload from a native Gemini request.
    This is used for direct Gemini API calls.
    """
    native_request["safetySettings"] = DEFAULT_SAFETY_SETTINGS
    
    if "generationConfig" not in native_request:
        native_request["generationConfig"] = {}
        
    # native_request["enableEnhancedCivicAnswers"] = False
    
    if "thinkingConfig" not in native_request["generationConfig"]:
        native_request["generationConfig"]["thinkingConfig"] = {}
    
    # Configure thinking based on model variant
    thinking_budget = get_thinking_budget(model_from_path)
    include_thoughts = should_include_thoughts(model_from_path)
    
    native_request["generationConfig"]["thinkingConfig"]["includeThoughts"] = include_thoughts
    if "thinkingBudget" in native_request["generationConfig"]["thinkingConfig"] and thinking_budget == -1:
        pass
    else:
        native_request["generationConfig"]["thinkingConfig"]["thinkingBudget"] = thinking_budget
    
    # Add Google Search grounding for search models
    if is_search_model(model_from_path):
        if "tools" not in native_request:
            native_request["tools"] = []
        # Add googleSearch tool if not already present
        if not any(tool.get("googleSearch") for tool in native_request["tools"]):
            native_request["tools"].append({"googleSearch": {}})
    
    return {
        "model": get_base_model_name(model_from_path),  # Use base model name for API call
        "request": native_request
    }
