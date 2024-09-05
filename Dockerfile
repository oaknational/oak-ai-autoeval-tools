# Use an official Python runtime as a parent image
FROM python:3

# Prevents Python from buffering output
ENV PYTHONUNBUFFERED True

# Expose the port that your app will run on
EXPOSE 8080

# Create a non-root user and group
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Create an application directory and set permissions
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy all necessary files to the container
COPY --chown=appuser:appgroup . ./

# Install the required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appgroup $APP_HOME

# Switch to the non-root user
USER appuser

# Command to run the app
CMD streamlit run --server.port $PORT --server.enableCORS false streamlit/Hello.py