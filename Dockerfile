FROM alpine:3.4
MAINTAINER think@hotmail.de

COPY darknet.sh /
RUN chmod +x darknet.sh \
 && apk add --update --no-cache \
    build-base curl \
    make \
    gcc \
    git \
    perl \
 && git clone https://github.com/pjreddie/darknet.git \
 && (cd /darknet && make && rm -rf scripts src results obj .git) \
 && curl -O http://pjreddie.com/media/files/extraction.weights \
 && apk del \
    build-base \
    ca-certificates \
    curl \
    gcc \
    git \
    libcurl \
    libgcc \
    libssh2 \
    pcre \
    perl \
    make \
    musl-dev \
 && rm -rf /var/cache/apk/*

WORKDIR "/darknet"
ENTRYPOINT ["/darknet.sh"]
CMD ["classifier", "predict", "cfg/imagenet22k.dataset", "cfg/extraction.cfg", "/extraction.weights"]
