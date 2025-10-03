"""
FastAPI web service for Hausa-English translation.
Provides REST API and web interface for NGO use.
"""

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
from datetime import datetime
import os

from translator import HausaEnglishTranslator, TranslationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hausa-English Translator API",
    description="AI-powered translation service for NGOs in Northern Nigeria",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize translator
translator = None


@app.on_event("startup")
async def startup_event():
    """Initialize translator on startup"""
    global translator
    try:
        logger.info("Initializing Hausa-English Translator...")
        translator = HausaEnglishTranslator()
        logger.info("Translator initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize translator: {e}")
        raise


# Pydantic models for API
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "ha"


class BatchTranslationRequest(BaseModel):
    texts: List[str]
    source_lang: str = "en"
    target_lang: str = "ha"


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    model_used: str
    timestamp: str


# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Main page with translation interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text between English and Hausa

    - **text**: Text to translate (required)
    - **source_lang**: Source language code (en/ha, default: en)
    - **target_lang**: Target language code (en/ha, default: ha)
    """
    try:
        if not translator:
            raise HTTPException(status_code=503, detail="Translator not initialized")

        result = translator.translate(
            text=request.text, source_lang=request.source_lang, target_lang=request.target_lang
        )

        return TranslationResponse(
            original_text=result.original_text,
            translated_text=result.translated_text,
            source_language=result.source_language,
            target_language=result.target_language,
            confidence_score=result.confidence_score,
            model_used=result.model_used,
            timestamp=result.timestamp.isoformat(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail="Translation failed")


@app.post("/translate/batch")
async def batch_translate(request: BatchTranslationRequest):
    """
    Translate multiple texts at once

    - **texts**: List of texts to translate
    - **source_lang**: Source language code (en/ha, default: en)
    - **target_lang**: Target language code (en/ha, default: ha)
    """
    try:
        if not translator:
            raise HTTPException(status_code=503, detail="Translator not initialized")

        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")

        results = translator.batch_translate(
            texts=request.texts, source_lang=request.source_lang, target_lang=request.target_lang
        )

        response_data = []
        for result in results:
            response_data.append(
                {
                    "original_text": result.original_text,
                    "translated_text": result.translated_text,
                    "source_language": result.source_language,
                    "target_language": result.target_language,
                    "confidence_score": result.confidence_score,
                    "model_used": result.model_used,
                    "timestamp": result.timestamp.isoformat(),
                }
            )

        return {"translations": response_data, "count": len(response_data)}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail="Batch translation failed")


@app.get("/history")
async def get_translation_history(limit: int = 50):
    """
    Get recent translation history

    - **limit**: Number of recent translations to return (max 100)
    """
    try:
        if not translator:
            raise HTTPException(status_code=503, detail="Translator not initialized")

        if limit > 100:
            limit = 100

        history = translator.get_translation_history(limit=limit)
        return {"history": history, "count": len(history)}

    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "languages": [
            {"code": "en", "name": "English", "native_name": "English"},
            {"code": "ha", "name": "Hausa", "native_name": "Harshen Hausa"},
        ]
    }


# Web interface endpoints
@app.post("/web/translate", response_class=HTMLResponse)
async def web_translate(
    request: Request,
    text: str = Form(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("ha"),
):
    """Web form translation endpoint"""
    try:
        if not translator:
            error = "Translator service is not available"
            return templates.TemplateResponse("index.html", {"request": request, "error": error})

        result = translator.translate(text, source_lang, target_lang)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "original_text": result.original_text,
                "translated_text": result.translated_text,
                "source_lang": result.source_language,
                "target_lang": result.target_language,
                "confidence": result.confidence_score,
                "model_used": result.model_used,
            },
        )

    except Exception as e:
        logger.error(f"Web translation error: {e}")
        error = f"Translation failed: {str(e)}"
        return templates.TemplateResponse("index.html", {"request": request, "error": error})


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About page"""
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs_page(request: Request):
    """API documentation page"""
    return templates.TemplateResponse("api_docs.html", {"request": request})


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info",
    )
