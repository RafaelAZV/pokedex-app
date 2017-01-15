from django.conf.urls import include, url
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView
from myapp import views

from django.contrib import admin

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^myapp/', include('myproject.myapp.urls')),
    url(r'^charmander', views.list, name='charmander'),
    url(r'^pikachu', views.list, name='pikachu'),
    url(r'^bulbasaur', views.list, name='bulbasaur'),
    url(r'^squirtle', views.list, name='squirtle'),
    url(r'^ekans', views.list, name='ekans'),
    url(r'^onix', views.list, name='onix'),
    url(r'^$', RedirectView.as_view(url='/myapp/list', permanent=True)),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
