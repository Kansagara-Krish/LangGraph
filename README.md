Emotion-Aware AI Chatbot using LangGraph & Gemini

This project implements an emotion-aware conversational AI using LangGraph and Google Gemini (2.0 Flash).
The chatbot intelligently classifies user messages as logical or emotional and dynamically routes the conversation to the appropriate agent.

🚀 Features

Automatic Message Classification

Classifies user input as:

logical → factual, informational, problem-solving

emotional → feelings, emotional support, personal struggles

Multi-Agent Architecture

Logical Agent: Provides concise, fact-based responses.

Therapist Agent: Responds with empathy, emotional validation, and reflective questions.

LangGraph Workflow

Uses a state-driven graph to control conversation flow.

Clean separation of concerns using nodes and conditional routing.

Structured Output Validation

Uses Pydantic to ensure reliable message classification.

Environment-Based Configuration

Secure API key management using .env.
