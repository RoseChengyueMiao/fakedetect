# fakedetect
#fake news detection application based on web single pattern

model weights is not included

- Python 3.8 以上
- Docker



```sh

$ curl localhost:8000/health
# 出力
# {
#   "health":"ok"
# }




$ curl localhost:8000/label






$ curl \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"data": "税に関しての発言について「個人的な思いで述べたもの」との見解を示した。6日の閣議後は、さらに玄葉光一郎外相が記者会見で、東京電力の電気料金値上げについて認めることがあってはならないと述べた。なお、同社ホームページによると、福島第一原子力発電所事故の「収束に向け、全力で取り組んでいるところで」「料金改定について言及できる段階では」ないとしている。"
}' \
    localhost:8000/predict
# 出力
# {
#   "prediction": [2]
# }
```

#Docker コンテナを停止

```sh
$ make stop
# 実行されるコマンド
# docker rm \
#   -f web_single_pattern
```

