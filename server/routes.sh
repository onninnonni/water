#!/bin/sh
# Keeps Onni's routes in the running Caddy config via the local admin API (re-run by a systemd
# user timer so they survive Caddy reloads):
#   /onni/seas/ws/*  -> Seven Seas WebSocket relay on 127.0.0.1:8798
#   /juha/water/*    -> static files in /srv/www/onni/juha-water (the original 2D simulator)
#   onnin.no         -> the game (/srv/www/onni/onnin), added only once DNS points here, so
#                       Caddy can obtain the certificate on the first try; www redirects to it.
#   /onni/onnin/*    -> 404, so the game is only served on onnin.no, not under pikos.org/onni/.
ADMIN=http://127.0.0.1:2019
SUB=$ADMIN/config/apps/http/servers/srv0/routes/0/handle/0/routes/0
TOP=$ADMIN/config/apps/http/servers/srv0/routes/0
IP=141.0.234.143
ensure() { # $1 id, $2 insert url, $3 json
  code=$(curl -s -o /dev/null -w '%{http_code}' "$ADMIN/id/$1")
  [ "$code" = "200" ] && return 0
  curl -s -o /dev/null -w "caddy route $1: HTTP %{http_code}\n" -X PUT "$2" -H 'Content-Type: application/json' -d "$3"
}
points_here() { r=$(dig +short "$1" A @ns1.dnsdomene.net | grep -E "^[0-9.]+$"); [ -n "$r" ] && ! echo "$r" | grep -vqx "$IP"; }
ensure onni-onnin-hide "$TOP" '{"@id":"onni-onnin-hide","match":[{"host":["pikos.org"],"path":["/onni/onnin","/onni/onnin/*","/onni/water","/onni/water/*"]}],"handle":[{"handler":"static_response","status_code":404,"body":"Not found. Seven Seas now lives at https://onnin.no"}],"terminal":true}'
ensure onni-seas-ws "$SUB" '{"@id":"onni-seas-ws","match":[{"path":["/onni/seas/ws/*"]}],"handle":[{"handler":"reverse_proxy","upstreams":[{"dial":"127.0.0.1:8798"}]}],"terminal":true}'
ensure onni-juha-water-redir "$SUB" '{"@id":"onni-juha-water-redir","match":[{"path":["/juha/water"]}],"handle":[{"handler":"static_response","status_code":308,"headers":{"Location":["/juha/water/"]}}],"terminal":true}'
ensure onni-juha-water "$SUB" '{"@id":"onni-juha-water","match":[{"path":["/juha/water/*"]}],"handle":[{"handler":"rewrite","strip_path_prefix":"/juha/water"},{"handler":"file_server","root":"/srv/www/onni/juha-water"}],"terminal":true}'
if points_here onnin.no; then
  ensure onni-onnin "$TOP" '{"@id":"onni-onnin","match":[{"host":["onnin.no"]}],"handle":[{"handler":"file_server","root":"/srv/www/onni/onnin"}],"terminal":true}'
fi
if points_here www.onnin.no; then
  ensure onni-onnin-www "$TOP" '{"@id":"onni-onnin-www","match":[{"host":["www.onnin.no"]}],"handle":[{"handler":"static_response","status_code":308,"headers":{"Location":["https://onnin.no{http.request.uri}"]}}],"terminal":true}'
fi
