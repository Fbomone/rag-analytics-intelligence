# src/data_loader.py
import os

class DataLoader:
    def __init__(self, data_path: str = "./data"):
        self.data_path = data_path
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create directories"""
        os.makedirs(os.path.join(self.data_path, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "documents"), exist_ok=True)
    
    def load_documents(self, folder: str = "documents") -> list:
        """Load all text documents"""
        docs_path = os.path.join(self.data_path, folder)
        documents = []
        
        # Create sample if empty
        if not os.path.exists(docs_path) or not os.listdir(docs_path):
            self.create_sample_documents(docs_path)
        
        for file in os.listdir(docs_path):
            if file.endswith('.txt'):
                with open(os.path.join(docs_path, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'filename': file,
                        'content': content
                    })
        
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def create_sample_documents(self, docs_path: str):
        """Create sample documents"""
        os.makedirs(docs_path, exist_ok=True)
        
        sample_docs = {
            'report_q3_2024.txt': """Q3 2024 Financial Report
=======================

Sales Performance:
- Total revenue: $2.5M (+18% YoY)
- Top products: Widget A (42%), Widget B (38%), Widget C (20%)
- Best performing region: EMEA (45% of sales)

Customer Metrics:
- New customers: 250 (+25% from Q2)
- Retention rate: 87%
- Average customer lifetime value: $45,000

Key Insights:
- Widget A strong demand
- EMEA highest growth
- Product C needs improvement

Recommendations:
- Increase Widget A production
- Expand EMEA sales team
- Review Product C pricing
""",
            'business_strategy.txt': """2024 Business Strategy
====================

Strategic Objectives:
1. Double market share in EMEA
2. Launch 3 new products
3. Achieve 40% gross margin

Product Roadmap:
- Widget A: Market leader
- Widget B: Enterprise focus
- Widget C: Premium market

Market Trends:
- AI/ML adoption +40% YoY
- Enterprise demand growing
- Cloud adoption accelerating
"""
        }
        
        for filename, content in sample_docs.items():
            with open(os.path.join(docs_path, filename), 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"✅ Created sample documents")