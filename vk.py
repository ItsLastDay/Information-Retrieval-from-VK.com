import datetime
import requests
from json import dumps, loads
import time

class Api(object):

    endpoint = ''

    def __init__(self, access_token):
        self.access_token = access_token

    def call(self, method, **params):
        request_params = params.copy()
        request_params["access_token"] = self.access_token
        request_params["v"] = "5.27"
        try:
            response = requests.get(self.endpoint.format(method=method), params=request_params)
            if response.status_code != 200:
                raise requests.exceptions.RequestException("Bad status code {}".format(response.status_code))
            return response.json()
        except requests.exceptions.RequestException as re:
            print("An API call failed with exception {}".format(re))
            raise

class VkApi(Api):

    endpoint = "https://api.vk.com/method/{method}"

    def get_comments(self, domain, post):
        try:
            d = self.call('wall.getComments', owner_id=domain, post_id=post, need_likes=1,\
                    offset=0, count=100, preview_length=0, sort='asc', extended=0)
            return d
        except:
            return loads('"error"')

    def get_group_info(self, domain):
        return self.call('groups.getById', group_id=domain, fields='members_count')

    def get_posts(self, domain):
        cnt = None
        offset = 0
        DF = 100
        out = open('./raw_vkapi_results/result_%s' % domain, 'w')

        while offset != cnt:
            time.sleep(1)
            count = DF if cnt == None else min(DF, cnt - offset)
            try:
                json = self.call("wall.get", domain=domain, offset=offset, count=count, filter='owner')

                for i in range(0 * len(json['response']['items'])):
                    domain_id = json['response']['items'][i]['owner_id']
                    post_id = json['response']['items'][i]['id']

                    if json['response']['items'][i]['comments']['count'] >= 80:
                        time.sleep(1.5)
                        json['response']['items'][i]['comments'] = self.get_comments(domain_id, post_id)
                
                out.write(dumps(json))
                out.write('\n')

                cnt = int(json.get('response').get('count'))
                print(domain, offset, cnt)
                offset += count
            except:
                time.sleep(10)

def main():
    '''
    https://oauth.vk.com/authorize?client_id=4227241&scope=9355263&redirect_uri=https://oauth.vk.com/blank.html&display=page&v=5.27&response_type=token
    '''
    token = open('cfg.txt', 'r').read().strip()
    domains = open('list.txt', 'r').read().strip().split('\n')
    api = VkApi(token)
    for domain in domains[:1]:
        api.get_posts(domain)


if __name__ == "__main__":
    main()
