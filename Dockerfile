FROM chronis10/teaching-ai-toolkit:arm64

COPY requirements.txt /app/requirements.txt

RUN pip install darknet
