[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/8IqmWj30)


## Overview  
**Streamlit dashboard** to analyze and query Reddit and 4chan data.  
It integrates the **GROQ model** for answering specific queries on this data in real-time.  

---

## How to Run  

### Option 1: Using Python Environment

1. **Install Dependencies**:  
   - Install required libraries using:  
     ```bash
     pip install -r requirements.txt
     ```

2. **Set Up Environment Variables**:  
   - Create a `.env` file in the root directory and add the following:
     ```env
     DB_NAME=crawler_db
     DB_USER=postgres
     DB_PASSWORD=password
     DB_HOST=localhost
     DB_PORT=5432
     GROQ_API_KEY=xx
     MODEL_NAME=llama-3.3-70b-versatile
     ```
   - To generate a GROQ API key, visit the GROQ platform's developer portal and follow their instructions.
      https://console.groq.com/

3. **Connect to DB**:  
    - Use the following command to connect to the database in the VM:  
      ```bash
      ssh -N -L <outport>:localhost:5432 -p 22 <username>@<ip>
      ```

4. **Run the Dashboard**:  
   - Start the Streamlit app with:  
     ```bash
     streamlit run main.py
     ```  

5. **Access the Dashboard**:  
   - Open the URL `http://localhost:8501` in your browser.  

---

### Option 2: Using Docker

1. **Build the Docker Image**:  
   - Build the Docker image using the provided Dockerfile:
     ```bash
     docker build -t streamlit-dashboard .
     ```

2. **Set Up Environment Variables**:  
   - Create a `.env` file in the root directory (if not already created) and include the following:
     ```env
     DB_NAME=crawler_db
     DB_USER=postgres
     DB_PASSWORD=password
     DB_HOST=localhost
     DB_PORT=5432
     GROQ_API_KEY=xx
     MODEL_NAME=llama-3.3-70b-versatile
     ```
   - To generate a GROQ API key, visit the GROQ platform's developer portal and follow their instructions.
     https://console.groq.com/
     
3. **Connect to DB**:  
    - Use the following command to connect to the database in the VM:  
      ```bash
      ssh -N -L <outport>:localhost:5432 -p 22 <username>@<ip>
      ```

4. **Run the Docker Container**:  
   - Start the container and link the `.env` file:
     ```bash
     docker run --env-file .env -p 8501:8501 streamlit-dashboard
     ```

5. **Access the Dashboard**:  
   - Open the URL `http://localhost:8501` in your browser.

---


