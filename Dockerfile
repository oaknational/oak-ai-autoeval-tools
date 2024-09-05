# Use an official Python runtime as a parent image
FROM python:3

# Prevents Python from buffering output
ENV PYTHONUNBUFFERED=True

# Expose the port that your app will run on
EXPOSE 8080

# Create a non-root user and group for running the app
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set the working directory
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Copy application code with secure permissions (owned by root, read-only for others)
COPY --chown=root:root --chmod=755 streamlit/ /app/streamlit/

# Copy requirements.txt with secure permissions
COPY --chown=root:root --chmod=644 requirements.txt /app/

# Create a logs directory for runtime logging or temp files
# Set ownership of the logs directory to appuser so it can write to it
RUN mkdir -p /app/streamlit/logs && \
    chown -R appuser:appgroup /app/streamlit/logs && \
    chmod -R 775 /app/streamlit/logs

# Install the required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-root user to run the application
USER appuser

# Command to run the app
CMD ["bash", "-c", "streamlit run streamlit/Hello.py --server.port=${PORT} --server.enableCORS=false"]
