# ⚖️ AI Legal Clause Risk Analyzer

## About the Project

I developed this project to make understanding legal documents and identifying potentially risky clauses easier.

Legal agreements often contain complex terms and conditions that can be difficult to understand, especially for people without a legal background. Going through lengthy documents manually can also take a lot of time.

To address this, I worked on an AI Legal Clause Risk Analyzer that analyzes legal clauses, identifies potential risks, and provides explanations to help users better understand the content.

## Why I Chose This Project

I wanted to explore how Natural Language Processing (NLP) can be applied to real-world problems, particularly in the legal domain.

Legal documents contain important information, but identifying risky terms and understanding their meaning can be challenging.

Through this project, I explored how NLP techniques can help process legal text and make important information easier to understand.

## What I Worked On

During this project, I worked on the following:

- **Dataset Preparation:** Worked with a legal clause dataset containing different types of legal clauses.
- **Text Processing:** Prepared and processed legal text to make it suitable for analysis.
- **Clause Analysis:** Developed functionality to analyze the content of legal clauses.
- **Risk Identification:** Worked on identifying potentially risky clauses based on their content and predefined criteria.
- **Risk Explanation:** Provided explanations to help users understand why a particular clause may require attention.
- **Application Development:** Created an interactive application using Python to make the analysis easier for users.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| NLP | Natural Language Processing |
| Text Processing | spaCy |
| Semantic Text Processing | Sentence-Transformers |
| Similarity Search | FAISS |
| Data Handling | Pandas |
| Application Development | Streamlit |
| Development Tools | VS Code, GitHub |

## How the Application Works

1. The user provides legal clause text for analysis.
2. The application processes the input text.
3. NLP techniques are used to analyze the clause content.
4. Potentially risky clauses are identified based on the implemented analysis logic.
5. The application provides relevant explanations to help users understand the identified risks.

## Project Structure

```text
AI-Legal-Clause-Risk-Analyzer/
│
├── README.md
├── app.py
├── build_vectorstore.py
├── clause_dataset.csv
├── requirements.txt
└── utils.py
```

## How to Run the Project

**Step 1: Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**Step 2: Navigate to the project directory**

```bash
cd AI-Legal-Clause-Risk-Analyzer
```

**Step 3: Install the required libraries**

```bash
pip install -r requirements.txt
```

**Step 4: Run the application**

```bash
streamlit run app.py
```

## What I Learned From This Project

This project helped me understand how Natural Language Processing can be used to work with domain-specific text such as legal documents.

I gained practical experience in text preprocessing, working with legal clause datasets, analyzing textual information, and developing an interactive application.

It also helped me understand how NLP techniques can be applied to solve practical problems beyond traditional text classification tasks.

## Future Improvements

I would like to improve this project by supporting complete legal document analysis, providing more detailed risk explanations, and expanding the dataset to cover different types of legal agreements.

---

## About Me

**Sindhu Sri**

B.Tech – Computer Science and Engineering (AI & ML)

Interested in Python Development, Natural Language Processing, Machine Learning, and Generative AI.

GitHub: https://github.com/SindhusriChattu
