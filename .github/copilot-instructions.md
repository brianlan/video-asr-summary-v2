## Overview

This project is handy tool that firstly extract audio from an input video, and then convert it into text using ASR service API, and finally the text are feed into a LLM to do a summarization. 

## Coding Guidelines
1. Strictly follow the TDD-First Approach when implementing a new feature
  - Write a failing test that simulates user or API interaction (`pytest`, etc.).
  - Write the minimal code to make the test pass.
  - Refactor to improve readability, maintainability, or modularity.
  - Repeat: one test → working feature → refactor cycle.
2. Continuous Integration
  - Automate tests and checks on every commit and pull request.
  - Use version control well: Descriptive commit messages. Small commits (1 change → 1 commit).
3. Code Quality
  - Follow **KISS**, **DRY**, **SOLID**, **YAGNI**.
  - Apply meaningful code comments; avoid explaining *what*, explain *why*.

## Testing Requirements
- **Mock External APIs**: Always mock OpenAI API calls to prevent real API usage during tests
- **Multi-Language Testing**: Test both custom language parameters and default English behavior
- **JSON Response Parsing**: Test both plain JSON and markdown-wrapped JSON responses from LLMs
- **Middleware Testing**: Test request/response logging middleware with and without database connectivity

## Important Instruction Reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Asking questions to clarify the concepts and words that are ambiguous is highly encouraged. If you see a flaw or problem, call it out and don't just follow them.