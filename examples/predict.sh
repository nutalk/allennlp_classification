python -m allennlp.service.server_simple \
    --archive-path /path/for/model/and/log/model.tar.gz \
    --predictor text_classifier \
    --include-package AllenFrame.classification_code \
    --title "文本分类" \
    --field-name sentence