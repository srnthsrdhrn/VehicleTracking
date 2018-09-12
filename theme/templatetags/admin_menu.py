import datetime

from django import template
from django.urls import reverse

register = template.Library()


@register.simple_tag
def current_time(format_string):
    return datetime.datetime.now().strftime(format_string)


# have to include icon code later
@register.inclusion_tag('theme/admin/menu.html')
def show_admin_menu(user):
    main_menu = [
        {
            "name": "Home",
            "childs": [
                {
                    "name": "Video list",
                    'url': reverse("list_video"),
                },
                # {
                #     "name": "Home",
                #     'url': reverse("productshome"),
                # }
            ]
        },


    ]

    filtered_menu = []
    user_permissions = user.get_all_permissions()
    for menu in main_menu:
        item = {'name': menu['name']}
        if "childs" in menu:
            item['childs'] = []
            for child in menu['childs']:
                if 'permission' in child:
                    if child['permission'] in user_permissions:
                        item["childs"].append(child)
                    else:
                        if isinstance(child['permission'], tuple):
                            for perm in child['permission']:
                                if perm in user_permissions:
                                    item["childs"].append(child)
                else:
                    item["childs"].append(child)
        filtered_menu.append(item)

    return {"menu": filtered_menu}
