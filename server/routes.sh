#!/bin/sh
# Keeps Onni's Seven Seas routes in the running Caddy config via the local admin API (re-run by a
# systemd user timer so they survive Caddy reloads):
#   /onni/seas/ws/*  -> WebSocket relay on 127.0.0.1:8798
#   /juha/water/*    -> static files in /srv/www/onni/juha-water (the original 2D simulator)
ADMIN=http://127.0.0.1:2019
SUB=$ADMIN/config/apps/http/servers/srv0/routes/0/handle/0/routes/0
ensure() { # $1 id, $2 json
  code=$(curl -s -o /dev/null -w '%{http_code}' "$ADMIN/id/$1")
  [ "$code" = "200" ] && return 0
  curl -s -o /dev/null -w "caddy route $1: HTTP %{http_code}\n" -X PUT "$SUB" -H 'Content-Type: application/json' -d "$2"
}
ensure onni-seas-ws '{"@id":"onni-seas-ws","match":[{"path":["/onni/seas/ws/*"]}],"handle":[{"handler":"reverse_proxy","upstreams":[{"dial":"127.0.0.1:8798"}]}],"terminal":true}'
ensure onni-juha-water-redir '{"@id":"onni-juha-water-redir","match":[{"path":["/juha/water"]}],"handle":[{"handler":"static_response","status_code":308,"headers":{"Location":["/juha/water/"]}}],"terminal":true}'
ensure onni-juha-water '{"@id":"onni-juha-water","match":[{"path":["/juha/water/*"]}],"handle":[{"handler":"rewrite","strip_path_prefix":"/juha/water"},{"handler":"file_server","root":"/srv/www/onni/juha-water"}],"terminal":true}'
