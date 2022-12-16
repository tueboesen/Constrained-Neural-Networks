search_dir=./scripts_to_run/
for entry in "$search_dir"/*.py
do
  a="$(basename $entry)"
  b="$(basename -s .py $a)"
  python -m scripts_to_run.$b
done
