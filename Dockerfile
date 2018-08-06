From pytorch/pytorch

WORKDIR /app

ADD . /app

RUN pip install -i https://pypi.douban.com/simple -r requirements.txt

EXPOSE 6666

CMD ["python", "app.py"]
