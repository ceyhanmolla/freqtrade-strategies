FROM freqtradeorg/freqtrade:stable

RUN pip install ta

ENTRYPOINT ["freqtrade"]
CMD ["trade"]