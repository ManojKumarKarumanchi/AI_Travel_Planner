# Limitations & Future Enhancements

## Current Scope

This is a functional proof-of-concept demonstrating a multi-agent travel planning workflow with human-in-the-loop approval. The system works end-to-end but intentionally prioritizes simplicity over production readiness.

## Current Limitations

- No DB, AUTH, and UI Integration 
- Just a placeholder code

### User Experience

* No user accounts or persistent profiles.
* No travel history, preferences, or personalization.
* Single-destination trips only.
* No itinerary sharing, calendar export, or PDF export.
* No streaming progress updates during planning.

### Infrastructure

* SQLite used for workflow persistence (not suitable for multi-instance deployments).
* No caching layer.
* No rate limiting or advanced monitoring.
* Limited retry and error-handling mechanisms.

### Testing & Security

* Limited automated testing.
* API keys stored in environment variables.
* No authentication or role-based access control.
* Development-focused security configuration.

## Recommended Next Steps

### Short Term

* PostgreSQL for persistence
* Redis caching
* Rate limiting
* Structured logging
* Unit and integration tests

### Medium Term

* User authentication
* User preferences and saved trips
* Weather API integration
* Calendar and PDF export
* Multi-destination trip support

### Long Term

* Personalized recommendations (RAG)
* Live flight and hotel integrations
* Booking and payment support
* Mobile application
* Collaborative trip planning

## Design Trade-offs

* **Simplicity over features** — Faster development and easier understanding.
* **Reliability over speed** — Structured outputs and fallback models improve success rates.
* **Maintainability over optimization** — Pure Python components reduce complexity at the cost of accuracy and performance.
