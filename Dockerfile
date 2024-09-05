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

# Copy application code with root ownership
COPY --chown=root:root streamlit/ /app/streamlit/

# Copy requirements.txt with root ownership
COPY --chown=root:root requirements.txt /app/

# Modify file permissions for security
RUN chmod -R 755 /app/streamlit && chmod 644 /app/requirements.txt

# Create a logs directory for runtime logging and set ownership to non-root user
RUN mkdir -p /app/streamlit/logs && \
    chown -R appuser:appgroup /app/streamlit/logs && \
    chmod -R 775 /app/streamlit/logs

# Install the required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-root user to run the application
USER appuser

# Command to run the app
CMD ["bash", "-c", "streamlit run streamlit/Hello.py --server.port=${PORT} --server.enableCORS=false"]
