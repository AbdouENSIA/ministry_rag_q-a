# Ministry Regulation Q&A System (Arabic Language)

A sophisticated Retrieval-Augmented Generation (RAG) system designed to provide accurate and accessible information about Ministry regulations through an Arabic-language chatbot interface. This is a proprietary system developed for exclusive use by the Ministry.

## Project Overview

This system implements an intelligent question-answering platform that processes and understands official ministry regulation documents using advanced RAG techniques. Users can interact with the system in Arabic through a user-friendly chatbot interface, receiving accurate, contextually grounded responses based on official documentation.

### Key Features

- Arabic language support with right-to-left (RTL) interface
- Intelligent document processing and indexing
- Advanced RAG pipeline for accurate information retrieval
- User-friendly chatbot interface
- Grounded responses based on official ministry regulations
- Scalable and maintainable architecture
- Secure and private deployment environment

## Architecture

The system implements a sophisticated RAG pipeline with the following components:

1. **Document Processing**
   - Arabic text processing and normalization
   - Secure document ingestion and indexing
   - Vector store management for efficient retrieval
   - Ministry-specific document validation

2. **RAG Pipeline**
   - Query Analysis: Processes Arabic queries for optimal retrieval
   - Document Retrieval: Fetches relevant documents from the vector store
   - Document Grading: Ranks and filters retrieved documents
   - Answer Generation: Generates responses in Arabic
   - Self-Reflection: Validates generated answers for accuracy and relevance

3. **User Interface**
   - Arabic chatbot interface
   - RTL layout support
   - Responsive design for various devices
   - Ministry-specific access controls

## Project Structure

```text
src/
├── nodes/              # LangGraph nodes
│   ├── query_analyzer.py
│   ├── retriever.py
│   ├── grader.py
│   ├── generator.py
│   └── arabic_processor.py
├── pipeline/           # RAG pipeline 
│   └── rag_pipeline.py
├── graph/              # Graph assembly
│   └── rag_graph.py
├── utils/              # Utility functions
├── config/             # Configuration
│   └── settings.py
└── types/              # Type definitions
    └── state.py
```

## Project Timeline

- Initial Prototype: April 16, 2024
- Focus: Ministry regulation documents from 2024
- Target Users: Ministry staff, students, teachers, and authorized personnel

## Project Team

- Course: Group Project
- Teacher: Abdelhakim Cheriet
- Student Group: Group 4
- Team Leader: Ainouche Abderahmane
- Team Members:
  - Moulahcene Riadh
  - Zamoum Abdelhakim
  - Guerroudj Abdennour
  - Mestouri Oussama
- Organization: Algerian Ministry of Education

## Future Enhancements

- Integration with additional ministry platforms
- Expansion to cover historical regulations
- Enhanced user experience features
- Performance optimizations
- Additional language support
- Advanced security features
- Ministry-specific analytics and reporting

## Confidentiality Notice

This project is proprietary and confidential. All rights reserved. Unauthorized copying, modification, distribution, or use of this project, via any medium, is strictly prohibited. Written permission must be obtained from the Ministry before any use or distribution of this system.
