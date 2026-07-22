<h2 id="publications" style="margin: 2px 0 10px;">Publications</h2>

<div class="publications">
<ol class="bibliography">
{% for link in site.data.publications.main %}
  <li>
    <div class="pub-row">
      <div style="position: relative; padding: 0 0 8px;">
        <div class="title">
          {% if link.conference_short %}
          <abbr class="badge" style="background-color: var(--global-theme-color); margin-right: 6px;">{{ link.conference_short }}</abbr>
          {% endif %}
          <a href="{{ link.paper }}">{{ link.title }}</a>
        </div>
        <div class="author">{{ link.authors }}</div>
        <div class="periodical"><em>{{ link.conference }}</em></div>
        <div class="links">
          {% if link.paper %}
          <a href="{{ link.paper }}" class="btn btn-sm z-depth-0" role="button" target="_blank" rel="noopener" style="font-size:12px;">Paper</a>
          {% endif %}
          {% if link.pdf %}
          <a href="{{ link.pdf }}" class="btn btn-sm z-depth-0" role="button" target="_blank" rel="noopener" style="font-size:12px;">PDF</a>
          {% endif %}
          {% if link.doi %}
          <a href="{{ link.doi }}" class="btn btn-sm z-depth-0" role="button" target="_blank" rel="noopener" style="font-size:12px;">DOI</a>
          {% endif %}
        </div>
      </div>
    </div>
  </li>
{% endfor %}
</ol>
</div>
