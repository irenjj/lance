#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

set -euo pipefail

tree=$(cargo tree --locked --offline -p lance --no-default-features \
    --edges normal,build,dev --prefix none "$@")
printf '%s\n' "$tree" | awk '
    $1 == "datafusion" {
        print "Unexpected DataFusion core: " $0 > "/dev/stderr"
        failed = 1
    }
    $1 ~ /^arrow($|-)/ || $1 == "parquet" {
        if ($2 != "v60.2.0") {
            print "Unexpected Arrow/Parquet version: " $0 > "/dev/stderr"
            failed = 1
        }
        seen[$1] = 1
    }
    $1 == "object_store" {
        if ($2 != "v0.14.2") {
            print "Unexpected object_store version: " $0 > "/dev/stderr"
            failed = 1
        }
        seen[$1] = 1
    }
    $1 ~ /^aws-sdk-/ || $1 == "aws-config" || $1 == "aws-credential-types" || $1 ~ /^lance-(namespace|geo|tools|datagen)($|-)/ {
        print "Unexpected cloud SDK or excluded Lance package: " $0 > "/dev/stderr"
        failed = 1
    }
    END {
        if (!seen["arrow-array"] || !seen["parquet"] || !seen["object_store"]) {
            print "Dependency tree is missing arrow-array, parquet, or object_store" > "/dev/stderr"
            failed = 1
        }
        if (failed) {
            exit 1
        }
        print "Lance read-only normal, build, and dev dependencies"
        print "DataFusion core packages: 0"
        print "Arrow and Parquet: 60.2.0 only"
        print "object_store: 0.14.2 only"
        print "AWS SDK and excluded Lance packages: 0"
    }
'
