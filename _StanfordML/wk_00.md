---
layout: pagecollection
title: Stanford Machine Learning
tagline: Contents
collection: StanfordML
---
{% include JB/setup %}


<table class="table condensed text-center">
  <tbody>
  {% for each_week in site.StanfordML %}
  {% if each_week.title == 'Stanford Machine Learning' %}
  <!-- Do nothing -->
  {% else %}
    <tr>
      <td><a href="{{ BASE_PATH }}{{ each_week.url }}">{{ each_week.title }}</a></td>
    </tr>
  {% endif %}
  {% endfor %}
  </tbody>
  </table>





