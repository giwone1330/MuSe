#!/bin/bash

start_dir="${1:-.}"

find "$start_dir" -path './.git' -prune -o -print | awk '
{
  gsub(/^\.\//, "");                # remove leading "./"
  n = split($0, parts, "/");        # split path into components
  path = "";
  for (i = 1; i <= n; i++) {
    path = (path ? path "/" : "") parts[i];
    if (!seen[path]) {
      indent = ""
      for (j = 1; j < i; j++) {
        indent = indent "│  "
      }
      if (i == n) {
        branch = (seen_parent[path]++ ? "├─ " : "└─ ")
        print indent branch parts[i]
      }
      seen[path] = 1
    }
  }
}'
