# Available modes
AVAILABLE_MODES = {
    "chat": "KnowledgeBase",
    "search": "search_mode"
}

# Metadata files for tracking processed PDFs
CHAT_METADATA_FILE = "./Documents/processed_pdfs_chat_mode.json"
SEARCH_METADATA_FILE = "./Documents/processed_pdfs_search_mode.json"

# ChromaDB paths
CHAT_DB_PATH = "./rag_database_chat_mode"
SEARCH_DB_PATH = "./rag_database_search_mode"

# ChromaDB collection names
CHAT_COLLECTION_NAME = "survival_knowledge_chat_mode"
SEARCH_COLLECTION_NAME = "survival_knowledge_search_mode"
