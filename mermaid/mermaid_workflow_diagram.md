# Mermaid Workflow Diagram old
```mermaid
---
config:
  theme: redux
  layout: dagre
---
flowchart TB
    A["User Input"] --> B["Planner"]
    B -- "sub-question 1" --> C["search, think, answer"]
    B -- "sub-question 2" --> D["search, think, answer"]
    B -- "sub-question 3" --> E["search, think, answer"]
    C --> F["reflection"]
    D --> G["reflection"]
    E --> H["reflection"]
    F --> I["Planner"]
    G --> I
    H --> I
    I -- false reflection --> J["Rewrite the question"]
    I -- true reflection --> K["Information list (answered QA pairs reflected TRUE)"]
    J --> L["search, think, answer"]
    L --> M["reflection"]
    M -- if true --> K
    K -- "Use sub-questions reflectred TRUE to update the question" --> J
    M -- if false--> J
    K --> O("Planner")
    O --> P("Final Answer")
```

# Mermaid Workflow Diagram
```mermaid
---
title: MARRAG Workflow
config:
  theme: base
  themeVariables:
    fontSize: 40px
    primaryColor: '#fff'
    secondaryColor: '#e6e6e6'
    primaryBorderColor: '#28253D'
    secondaryBorderColor: '#000000'
---
flowchart LR
    A["User<br>Input"] --> B["Planner 1"]
    B --> C["sub-question 1"] & D["sub-question 2"] & E["sub-question 3"]
    C --> G["Reflector"]
    D --> G
    E --> G
    G -- Reflectted FALSE --> H["Planner 2 (generate new sub-questions)"]
    G -- Reflectted TRUE --> I["True Reflection List"]
    H -- "new sub-questions" --> J["Reflector"]
    J -- Reflectted TRUE --> I
    I -- Contextual Support --> H
    J -- Reflectted FALSE --> H
    I --> K["Planner 3"]
    K --> L["Final</br>Answer"]

    style B fill:#FFE0B2
    style H fill:#C8E6C9
    style K fill:#BBDEFB
```

# Planner1 (sub-question generation) diagram
```mermaid
---
config:
  theme: base
  themeVariables:
    fontSize: 40px
    primaryColor: '#fff'
    secondaryColor: '#e6e6e6'
    primaryBorderColor: '#28253D'
    secondaryBorderColor: '#000000'
---
flowchart LR
  A["User Input Question"] --> B["Planner 1"]
  B --> C["Think, analyze the question"]
  C --> D["sub-question 1"]
  C --> E["sub-question 2"]
  C --> F["sub-question 3"]

  style B fill:#FFE0B2
```

# Planner2 (sub-question re-generation) diagram
```mermaid
---
config:
  theme: base
  themeVariables:
    fontSize: 40px
    primaryColor: '#fff'
    secondaryColor: '#e6e6e6'
    primaryBorderColor: '#28253D'
    secondaryBorderColor: '#000000'
---
flowchart LR
  A["User Input Question"] --> P2["Planner 2"]
  B["True Reflection List"] --> P2
  P2 -- Determine --> C["'What information do I still need to answer the question'"]
  C --> D["new sub-question 1"]
  C --> E["new sub-question 2"]
  C --> F["new sub-question 3"]

  style P2 fill:#C8E6C9
```


# Planner3 (Final answer analyser) diagram
```mermaid
---
config:
  theme: base
  themeVariables:
    fontSize: 40px
    primaryColor: '#fff'
    secondaryColor: '#e6e6e6'
    primaryBorderColor: '#28253D'
    secondaryBorderColor: '#000000'
---
flowchart LR
  A["User Input Question"] --> P3["Planner 3"]
  B["True Reflection List"] --> P3
  P3 --> C["Reasoning over TRL and input question"]
  C --> D["Final Answer"]


  style P3 fill:#BBDEFB
```


# Reflector Generation workflow diagram
```mermaid
---
config:
  theme: base
  themeVariables:
    fontSize: 40px
    primaryColor: '#fff'
    secondaryColor: '#e6e6e6'
    primaryBorderColor: '#28253D'
    secondaryBorderColor: '#000000'
---
flowchart LR
    A["Input question"] --> Ref["Reflector Model"]
    B["Search Tool"] --> Ref
    Ref --> C["Reasoning"]
    C --> D["Predicted Answer"]
    D --> E["self-reflection reasoning"]
    E --> F["self-reflection verification"]
    F --> G["True"] & H["False"]

    style Ref fill:#E1BEE7
    style D fill:#FFE0B2
    style F fill:#FFCDD2
```
