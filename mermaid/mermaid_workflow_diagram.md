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
```

# Planner2 (sub-question re-generation) diagram
```mermaid
```


# Planner3 (Final answer analyser) diagram
```mermaid
```


# Reflector Generation workflow diagram
```mermaid
```
