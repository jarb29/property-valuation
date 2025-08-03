"""
FastAPI application.

This module defines the FastAPI application and its configuration.
"""

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
import logging
import time


from src.api.auth import get_api_key
from src.api.endpoints import router as api_router
from src.api.middleware import LoggingMiddleware
from src.api.logging import setup_logging
from src.config import API_DEBUG

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app with professional documentation
app = FastAPI(
    title="Property Valuation API",
    description="""
    ## Enterprise Real Estate Valuation Platform

    Professional API service for accurate property valuations in the Chilean real estate market.
    Built with machine learning algorithms for enterprise integration.

    ### Core Features
    - **Machine Learning**: Advanced models with validated accuracy
    - **High Performance**: Sub-second response times
    - **Batch Processing**: Bulk property evaluation
    - **Secure**: API key authentication with audit logging

    ### Property Types
    **Departamentos** | **Casas** | **Oficinas**

    ### Coverage
    Metropolitan Santiago including Las Condes, Vitacura, Providencia, and surrounding areas.

    ### Authentication
    ```http
    X-API-Key: default_api_key
    ```

    **Support**: enterprise@property-friends.cl
    """,
    version="2.1.0",
    terms_of_service="https://property-friends.cl/enterprise/terms",
    contact={
        "name": "Enterprise API Support",
        "url": "https://property-friends.cl/enterprise/support",
        "email": "enterprise@property-friends.cl",
    },
    license_info={
        "name": "Enterprise License",
        "url": "https://property-friends.cl/enterprise/license",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={
        "deepLinking": True,
        "displayRequestDuration": True,
        "docExpansion": "none",
        "operationsSorter": "method",
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True
    },
    openapi_tags=[
        {
            "name": "Property Valuation",
            "description": "Core prediction endpoints for property valuation services",
        },
        {
            "name": "Model Management",
            "description": "Model information and metadata endpoints",
        },
        {
            "name": "System Monitoring",
            "description": "Health monitoring and system status endpoints",
        },
    ],
    debug=API_DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Custom Swagger UI with professional styling
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "none",
            "operationsSorter": "method",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True
        },
        swagger_ui_init_oauth={
            "usePkceWithAuthorizationCodeGrant": True,
        },
        swagger_favicon_url="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' fill='%233b82f6'/><text y='70' font-size='60' fill='white' text-anchor='middle' x='50'>PF</text></svg>"
    )

# Include API router with authentication
app.include_router(
    api_router,
    dependencies=[Depends(get_api_key)]
)

# Custom documentation styling
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Custom landing page with enhanced styling.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Property Valuation API - Enterprise Platform</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
                background: #f8fafc;
                margin: 0;
                padding: 0;
                min-height: 100vh;
                color: #334155;
            }
            .header {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                color: white;
                padding: 2rem 0;
                text-align: center;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
            }
            .logo {
                width: 60px;
                height: 60px;
                background: #3b82f6;
                border-radius: 12px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
                margin-bottom: 1rem;
            }
            h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0 0 0.5rem 0;
                letter-spacing: -0.025em;
            }
            .subtitle {
                font-size: 1.25rem;
                opacity: 0.9;
                margin-bottom: 2rem;
                font-weight: 400;
            }
            .main-content {
                padding: 3rem 0;
            }
            .buttons {
                display: flex;
                gap: 1rem;
                justify-content: center;
                flex-wrap: wrap;
                margin-bottom: 3rem;
            }
            .btn {
                padding: 0.875rem 2rem;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                text-decoration: none;
                transition: all 0.2s ease;
                font-weight: 600;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
            }
            .btn-primary {
                background: #3b82f6;
                color: white;
            }
            .btn-secondary {
                background: white;
                color: #475569;
                border: 1px solid #e2e8f0;
            }
            .btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .btn-primary:hover {
                background: #2563eb;
            }
            .features {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 1rem;
                margin-bottom: 3rem;
            }
            @media (max-width: 768px) {
                .features {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
            @media (max-width: 480px) {
                .features {
                    grid-template-columns: 1fr;
                }
            }
            .feature {
                background: white;
                padding: 1.5rem 1rem;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
                text-align: center;
            }
            .feature-icon {
                font-size: 1.5rem;
                margin-bottom: 0.5rem;
            }
            .feature h3 {
                font-size: 0.875rem;
                font-weight: 600;
                margin: 0 0 0.25rem 0;
                color: #1e293b;
            }
            .feature p {
                color: #64748b;
                margin: 0;
                font-size: 0.75rem;
                line-height: 1.4;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <div class="logo">PF</div>
                <h1>Property Valuation API</h1>
                <p class="subtitle">Enterprise Real Estate Intelligence Platform</p>
            </div>
        </div>
        <div class="main-content">
            <div class="container">

                <div class="buttons">
                    <a href="/docs" class="btn btn-primary">API Documentation</a>
                    <a href="/redoc" class="btn btn-secondary">Technical Reference</a>
                    <a href="/health" class="btn btn-secondary">System Status</a>
                </div>

                <div style="background: white; padding: 2rem; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 3rem;">
                    <h3 style="margin-top: 0; color: #1e293b; font-size: 1.25rem; font-weight: 600;">API Authentication</h3>
                    <p style="margin-bottom: 1rem; color: #64748b;">Enterprise API access requires authentication. Use the following key for testing:</p>
                    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; font-family: 'SF Mono', Monaco, monospace; font-size: 0.875rem; border: 1px solid #e2e8f0; position: relative; display: flex; align-items: center; justify-content: space-between;">
                        <span id="apiKey" style="color: #1e293b;">default_api_key</span>
                        <button onclick="copyApiKey()" style="background: #3b82f6; color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.875rem; font-weight: 500;">Copy</button>
                    </div>
                    <p style="margin-top: 1rem; font-size: 0.875rem; color: #64748b;">Include as header: <code style="background: #f1f5f9; padding: 0.25rem 0.5rem; border-radius: 4px; font-family: monospace;">X-API-Key: default_api_key</code></p>
                </div>

                <div class="features">
                    <div class="feature">
                        <div class="feature-icon">🤖</div>
                        <h3>AI Intelligence</h3>
                        <p>Advanced ML models with validated accuracy</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">⚡</div>
                        <h3>High Performance</h3>
                        <p>Sub-second response times</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🔒</div>
                        <h3>Enterprise Security</h3>
                        <p>API key auth with audit logging</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">📊</div>
                        <h3>Batch Processing</h3>
                        <p>Bulk property evaluation capabilities</p>
                    </div>
                </div>
            </div>
        </div>
        <script>
            function copyApiKey() {
                const apiKey = document.getElementById('apiKey').textContent.trim();
                const button = document.querySelector('button[onclick="copyApiKey()"]');
                const originalText = button.textContent;
                
                // Try modern clipboard API first
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(apiKey).then(function() {
                        button.textContent = '✓ Copied!';
                        button.style.background = '#10b981';
                        showToast('API key copied to clipboard!');
                        
                        setTimeout(function() {
                            button.textContent = originalText;
                            button.style.background = '#3b82f6';
                        }, 2000);
                    }).catch(function() {
                        fallbackCopy(apiKey, button, originalText);
                    });
                } else {
                    fallbackCopy(apiKey, button, originalText);
                }
            }
            
            function fallbackCopy(text, button, originalText) {
                const textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                
                try {
                    textarea.select();
                    textarea.setSelectionRange(0, 99999);
                    const successful = document.execCommand('copy');
                    
                    if (successful) {
                        button.textContent = '✓ Copied!';
                        button.style.background = '#10b981';
                        showToast('API key copied to clipboard!');
                    } else {
                        button.textContent = 'Select & Copy';
                        button.style.background = '#f59e0b';
                        showToast('Please select and copy: ' + text, 'warning');
                    }
                } catch (err) {
                    button.textContent = 'Select & Copy';
                    button.style.background = '#f59e0b';
                    showToast('Please select and copy: ' + text, 'warning');
                } finally {
                    document.body.removeChild(textarea);
                    
                    setTimeout(function() {
                        button.textContent = originalText;
                        button.style.background = '#3b82f6';
                    }, 3000);
                }
            }

            function showToast(message, type = 'success') {
                const colors = {
                    success: '#10b981',
                    error: '#ef4444',
                    warning: '#f59e0b'
                };
                
                const toast = document.createElement('div');
                toast.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: ${colors[type] || colors.success};
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    font-weight: 500;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                    z-index: 10000;
                    transform: translateX(100%);
                    transition: transform 0.3s ease;
                    max-width: 300px;
                    word-wrap: break-word;
                `;
                toast.textContent = message;
                
                document.body.appendChild(toast);
                
                setTimeout(() => {
                    toast.style.transform = 'translateX(0)';
                }, 100);
                
                setTimeout(() => {
                    toast.style.transform = 'translateX(100%)';
                    setTimeout(() => {
                        if (document.body.contains(toast)) {
                            document.body.removeChild(toast);
                        }
                    }, 300);
                }, 4000);
            }
        </script>
    </body>
    </html>
    """

# Health check endpoint (no authentication required)
@app.get("/health", tags=["❤️ Health & Status"])
async def health_check():
    """
    🩺 **System Health Check**

    Verify that the API service is running properly and all components are operational.

    **Perfect for:**
    - 🔄 Load balancer health checks
    - 📊 Monitoring systems
    - 🚀 Deployment verification
    - 🧪 Integration testing

    **Returns:**
    - ✅ **status**: Service operational status
    - 🕐 **timestamp**: Current server time
    - 🏷️ **version**: API version number
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": app.version,
        "service": "Property-Friends Real Estate Valuation API",
        "uptime": "operational"
    }

# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle all unhandled exceptions.

    Args:
        request (Request): The request that caused the exception.
        exc (Exception): The exception that was raised.

    Returns:
        JSONResponse: A JSON response with error information.
    """
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if API_DEBUG else "An unexpected error occurred"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Perform actions when the application starts.
    """
    logger.info("API starting up")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Perform actions when the application shuts down.
    """
    logger.info("API shutting down")
