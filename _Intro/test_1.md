---
layout: pagecollection
title: test_1
tagline: The Collection
collection: Intro
---
{% include JB/setup %}


<!-- <ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul> -->

<table class="table condensed text-center">
  <tbody>
  {% for each_week in site.Intro %}
    <h2>{{ each_week.title }}</h2>
  {% endfor %}
  </tbody>
  </table>




