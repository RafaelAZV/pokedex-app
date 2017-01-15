# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from subprocess import call
from myproject.myapp.models import Document
from myproject.myapp.forms import DocumentForm
from tpICV3 import *
import pdb

def list(request):
    # Handle file upload
    pokemonData = []
    labels = []

    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
            print request.FILES['docfile'].name
            getFeatures('bulbasaur', pokemonData, labels)
            getFeatures('charmander', pokemonData, labels)
            getFeatures('pikachu', pokemonData, labels)
            getFeatures('squirtle', pokemonData, labels)
            getFeatures('ekans', pokemonData, labels)
            getFeatures('onix', pokemonData, labels)

            svms = getEnsemble(pokemonData, labels)
            votes = classifyQuery(svms, request.FILES['docfile'].name)
            page = getPokemonPage(votes)

            # Redirect to the document list after POST
            return render(request, page, {})
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page

    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'list.html',
        {'form': form}
    )
