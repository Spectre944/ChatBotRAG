
import ssl
import requests
from requests.adapters import HTTPAdapter


class ProxySSLAdapter(HTTPAdapter):
    def proxy_manager_for(self, proxy, **kwargs):
        context = ssl.create_default_context()

        # Только диагностический тест:
        # отключаем строгую проверку формата сертификатов X.509.
        # Проверка доверенной цепочки и имени сайта остаётся включённой.
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            context.verify_flags &= ~ssl.VERIFY_X509_STRICT

        kwargs["ssl_context"] = context
        return super().proxy_manager_for(proxy, **kwargs)


session = requests.Session()
session.mount("https://", ProxySSLAdapter())

url = "https://huggingface.co/intfloat/multilingual-e5-base/resolve/main/modules.json"

try:
    response = session.get(url, timeout=20)
    print("Status:", response.status_code)
    print(response.text[:300])
except requests.RequestException as exc:
    print(type(exc).__name__, exc)
