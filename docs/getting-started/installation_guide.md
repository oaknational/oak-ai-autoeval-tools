# AutoEval Getting Started: Installation Guide

AutoEval utilizes the Streamlit open-source Python framework. A PostgreSQL database is required and the tool is configured to make API calls to OpenAI's LLMs. All AutoEval scripts are contained in the `streamlit` directory of this repository.  

To install AutoEval, follow the following steps:  

1. Clone the repository to run AutoEval locally without making any modifications. Fork the repository if you plan to make changes and experiment with the codebase.

2. Install the requirements.txt file.

    ```bash
    pip install -r streamlit/requirements.txt
    ```

3. Install PostgreSQL.

    Windows:
    - Download the installer for the latest version of PostgreSQL from the [PostgreSQL Downloads](https://www.postgresql.org/download/windows/) page.
    - Run the downloaded installer file.
    - Follow the setup wizard steps.

    macOS:

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install postgresql
    initdb /usr/local/var/postgres
    brew services start postgresql
    ```

    Ubuntu:

    ```bash
    sudo apt-get update
    sudo apt-get install postgresql postgresql-contrib
    ```

    *You can now use the psql command line tool or pgAdmin to manage your databases.*

4. Create a database and a user.

    ```bash
    psql -U postgres
    CREATE DATABASE mydatabase;
    CREATE USER myusername WITH PASSWORD 'mypassword';
    GRANT ALL PRIVILEGES ON DATABASE mydatabase TO myusername;
    \q
    ```

5. Use `.env.example` as a template to create a .env file for configuration settings, API keys, and database credentials.

6. Run the following command in the terminal to create tables and insert some placeholder data into your database.

    ```bash
    python streamlit/db_operations.py
    ```

7. Run the following command in the terminal to open the app in a new tab in your default browser.

    ```bash
    streamlit run streamlit/Hello.py
    ```
