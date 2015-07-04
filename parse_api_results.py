import json
from vk import VkApi

output = []

api = VkApi('')
domains = open('list.txt', 'r').read().strip().split('\n')
for domain in domains:
    info = api.get_group_info(domain)
    gname = info['response'][0]['name']
    gphoto = info['response'][0]['photo_50']
    gmembers = info['response'][0]['members_count']

    data = open('./raw_vkapi_results/result_%s' % domain, 'r').read().strip().split('\n')

    for p in data:
        pack = json.loads(p)['response']['items']

        for row in pack:
            cur = dict()
            cur['gn'] = gname # group name
            cur['gc'] = gmembers # group member count
            cur['gp'] = gphoto # group photo link

            cur['c'] = row['comments'].get('count', 80) # number of comments
            cur['l'] = row['likes']['count'] # number of likes 
            cur['r'] = row['reposts']['count'] # number of resposts

            cur['d'] = row['date'] # date (unix-time)
            cur['id'] = row['id'] # post id
            cur['t'] = row['text'] # text
            cur['gid'] = row['owner_id'] # owner id


            for att in row.get('attachments', []):
                if 'photo' in att and 'photo_807' in att['photo']:
                    cur['pp'] = att['photo']['photo_807'] # link to photo

                if 'link' in att:
                    cur['lte'] = att['link']['description'] # referenced text
                    cur['lti'] = att['link']['title'] # referenced title


            output.append(cur)



s = json.dumps(output, separators=(',',':'))

out = open('parsed_vkapi.txt', 'w')
out.write(s)
out.close()
