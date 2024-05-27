FROM python:3.6

RUN pip3 install -U spacy==1.9.0 && \
    python3 -m spacy download en && \
    pip3 install -U nltk && \
    git clone https://github.com/chrisjbryant/errant.git -b bea2019st

ADD evaluate.sh .

CMD ["sh", "evaluate.sh"]
