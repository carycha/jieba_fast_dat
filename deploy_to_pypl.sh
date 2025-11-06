uv build
auditwheel repair dist/*.whl -w dist/ && rm dist/*-linux_x86_64.whl
uv run twine check dist/*
uv run twine upload --repository testpypi dist/*
uv run twine upload dist/*
