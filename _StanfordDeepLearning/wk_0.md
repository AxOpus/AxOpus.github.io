---
layout: pagecollection
title: Stanford Deep Learning
tagline: Contents
collection: StanfordDeepLearning
---
{% include JB/setup %}


<table class="table condensed text-center">
  <tbody>
  {% for each_week in site.StanfordDeepLearning %}
  {% if each_week.title == 'Stanford Deep Learning' %}
  <!-- Do nothing -->
  {% else %}
    <tr>
      <td><a href="{{ BASE_PATH }}{{ each_week.url }}">{{ each_week.title }}</a></td>
    </tr>
  {% endif %}
  {% endfor %}
  </tbody>
  </table>





