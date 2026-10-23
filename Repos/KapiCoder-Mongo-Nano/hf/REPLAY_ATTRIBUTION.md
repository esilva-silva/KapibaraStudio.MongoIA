# Replay attribution / Atribuição do replay

A small share of the fine-tuning tokens (about 12%) is generic JavaScript/TypeScript taken from npm packages already installed on the
training machine, used only as training text ("replay") to limit forgetting of generic code. The code is **not redistributed** with this model.
All packages below are under permissive licenses (MIT, Apache-2.0, BSD-2/3-Clause, ISC, some dual-licensed with another permissive option); original headers were kept in the training text.
Packages used only for evaluation are not listed. Source table: the project's `data/replay/v1/LICENSES.md`.

Uma parcela pequena dos tokens de ajuste fino (cerca de 12%) é código JavaScript/TypeScript genérico de pacotes npm, usado só como texto de treino
(replay) para limitar o esquecimento de código genérico. O código **não é redistribuído** com este modelo.

Packages used in training: **555**.

| License | Packages |
|---|---:|
| MIT | 402 |
| ISC | 94 |
| Apache-2.0 | 22 |
| BSD-3-Clause | 21 |
| BSD-2-Clause | 11 |
| (Apache-2.0 AND BSD-3-Clause) | 1 |
| (BSD-2-Clause OR MIT OR Apache-2.0) | 1 |
| (MIT OR Apache-2.0) | 1 |
| (MIT OR WTFPL) | 1 |
| Apache 2.0 | 1 |

| Package | Version | License | Copyright line |
|---|---|---|---|
| 7zip-bin | 5.2.0 | MIT | Copyright (c) 2016 Vladimir Krivosheev |
| @ampproject/remapping | 2.2.1 | Apache-2.0 | - |
| @azure/abort-controller | 2.1.2 | MIT | Copyright (c) 2020 Microsoft |
| @azure/core-auth | 1.10.1 | MIT | Copyright (c) Microsoft Corporation. |
| @azure/core-lro | 2.7.2 | MIT | Copyright (c) 2020 Microsoft |
| @azure/core-paging | 1.6.2 | MIT | Copyright (c) 2020 Microsoft |
| @azure/core-rest-pipeline | 1.24.0 | MIT | Copyright (c) Microsoft Corporation. |
| @azure/core-tracing | 1.3.1 | MIT | Copyright (c) Microsoft Corporation. |
| @azure/identity | 4.13.1 | MIT | Copyright (c) Microsoft Corporation. |
| @azure/logger | 1.3.0 | MIT | Copyright (c) Microsoft Corporation. |
| @azure/msal-browser | 5.13.0 | MIT | Copyright (c) Microsoft Corporation. All rights reserved. |
| @azure/msal-common | 16.8.0 | MIT | Copyright (c) Microsoft Corporation. All rights reserved. |
| @azure/msal-node | 5.2.4 | MIT | Copyright (c) 2020 Microsoft |
| @azure/msal-node-runtime | 0.20.6 | MIT | - |
| @azure/static-web-apps-cli | 2.0.9 | MIT | Copyright (c) 2021 Microsoft |
| @babel/cli | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/code-frame | 7.27.1 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/generator | 7.23.6 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-annotate-as-pure | 7.22.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-create-class-features-plugin | 7.23.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-create-regexp-features-plugin | 7.22.15 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-define-polyfill-provider | 0.4.4 | MIT | Copyright (c) 2014-present Nicolò Ribaudo and other contributors |
| @babel/helper-environment-visitor | 7.22.20 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-function-name | 7.23.0 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-hoist-variables | 7.22.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-member-expression-to-functions | 7.23.0 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-optimise-call-expression | 7.22.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-plugin-utils | 7.22.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-remap-async-to-generator | 7.22.20 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-replace-supers | 7.22.20 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-simple-access | 7.22.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-skip-transparent-expression-wrappers | 7.22.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-split-export-declaration | 7.22.6 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-string-parser | 7.27.1 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-validator-identifier | 7.28.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-validator-option | 7.23.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helper-wrap-function | 7.22.20 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/helpers | 7.28.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/parser | 7.28.5 | MIT | Copyright (C) 2012-2014 by various contributors (see AUTHORS) |
| @babel/plugin-bugfix-safari-id-destructuring-collision-in-function-expression | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-bugfix-v8-spread-parameters-in-optional-chaining | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-bugfix-v8-static-class-fields-redefine-readonly | 7.23.7 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-proposal-private-property-in-object | 7.21.0-placeholder-for-preset-env.2 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-async-generators | 7.8.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-class-properties | 7.12.13 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-class-static-block | 7.14.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-dynamic-import | 7.8.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-export-namespace-from | 7.8.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-import-assertions | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-import-attributes | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-import-meta | 7.10.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-json-strings | 7.8.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-logical-assignment-operators | 7.10.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-nullish-coalescing-operator | 7.8.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-numeric-separator | 7.10.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-object-rest-spread | 7.8.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-optional-catch-binding | 7.8.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-optional-chaining | 7.8.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-private-property-in-object | 7.14.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-top-level-await | 7.14.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-syntax-unicode-sets-regex | 7.18.6 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-arrow-functions | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-async-generator-functions | 7.23.7 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-async-to-generator | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-block-scoped-functions | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-class-properties | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-class-static-block | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-classes | 7.23.8 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-computed-properties | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-destructuring | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-dotall-regex | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-duplicate-keys | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-dynamic-import | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-exponentiation-operator | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-export-namespace-from | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-for-of | 7.23.6 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-function-name | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-json-strings | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-literals | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-logical-assignment-operators | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-member-expression-literals | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-modules-amd | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-modules-commonjs | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-modules-systemjs | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-modules-umd | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-named-capturing-groups-regex | 7.22.5 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-new-target | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-nullish-coalescing-operator | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-numeric-separator | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-object-rest-spread | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-object-super | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-optional-catch-binding | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-optional-chaining | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-parameters | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-private-methods | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-private-property-in-object | 7.23.4 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-property-literals | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-regenerator | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-reserved-words | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-shorthand-properties | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-spread | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-sticky-regex | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-template-literals | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-typeof-symbol | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-unicode-escapes | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-unicode-property-regex | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-unicode-regex | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/plugin-transform-unicode-sets-regex | 7.23.3 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/preset-env | 7.23.8 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/preset-modules | 0.1.6-no-external-plugins | MIT | Copyright (c) 2020 Babel |
| @babel/regjsgen | 0.8.0 | MIT | Copyright 2014-2020 Benjamin Tan <https://ofcr.se/> |
| @babel/runtime | 7.29.7 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @babel/traverse | 7.23.7 | MIT | Copyright (c) 2014-present Sebastian McKenzie and other contributors |
| @bufbuild/protobuf | 2.13.0 | (Apache-2.0 AND BSD-3-Clause) | - |
| @gar/promise-retry | 1.0.3 | MIT | Copyright (c) 2011 Tim Koschützki, Felix Geisendörfer |
| @grpc/grpc-js | 1.9.0 | Apache-2.0 | - |
| @hapi/topo | 5.1.0 | BSD-3-Clause | Copyright (c) 2012-2020, Sideway Inc, and project contributors |
| @img/colour | 1.1.0 | MIT | Copyright (c) 2012 Heather Arthur |
| @isaacs/cliui | 8.0.2 | ISC | Copyright (c) 2015, Contributors |
| @isaacs/string-locale-compare | 1.1.0 | ISC | Copyright (c) Isaac Z. Schlueter |
| @jridgewell/gen-mapping | 0.3.3 | MIT | Copyright 2022 Justin Ridgewell |
| @jridgewell/resolve-uri | 3.1.1 | MIT | Copyright 2019 Justin Ridgewell |
| @jridgewell/set-array | 1.1.2 | MIT | Copyright 2022 Justin Ridgewell |
| @jridgewell/source-map | 0.3.5 | MIT | Copyright 2019 Justin Ridgewell |
| @jridgewell/sourcemap-codec | 1.4.15 | MIT | Copyright (c) 2015 Rich Harris |
| @jridgewell/trace-mapping | 0.3.31 | MIT | Copyright 2024 Justin Ridgewell |
| @lmstudio/lms-isomorphic | 0.4.6 | Apache-2.0 | Copyright (c) 2025 Element Labs Inc |
| @npmcli/agent | 4.0.0 | ISC | - |
| @npmcli/arborist | 9.4.3 | ISC | Copyright npm, Inc. |
| @npmcli/config | 10.8.1 | ISC | Copyright (c) npm, Inc. |
| @npmcli/fs | 5.0.0 | ISC | Copyright npm, Inc. |
| @npmcli/git | 7.0.2 | ISC | Copyright (c) npm, Inc. |
| @npmcli/installed-package-contents | 4.0.0 | ISC | Copyright (c) npm, Inc. |
| @npmcli/map-workspaces | 5.0.3 | ISC | Copyright npm, Inc. |
| @npmcli/node-gyp | 5.0.0 | ISC | Copyright (c) 2023 by GitHub Inc. |
| @npmcli/package-json | 7.0.5 | ISC | Copyright GitHub Inc. |
| @npmcli/query | 5.0.0 | ISC | Copyright npm, Inc. |
| @npmcli/redact | 4.0.0 | ISC | Copyright (c) 2024 npm |
| @npmcli/run-script | 10.0.4 | ISC | Copyright (c) npm, Inc. |
| @picovoice/pvrecorder-node | 1.2.9 | Apache-2.0 | - |
| @pnpm/config.env-replace | 1.1.0 | MIT | - |
| @pnpm/network.ca-file | 1.0.2 | MIT | - |
| @pnpm/npm-conf | 3.0.3 | MIT | Copyright (c) Kevin Mårtensson (github.com/kevva) |
| @pondwader/socks5-server | 1.0.10 | MIT | Copyright (c) 2024 PondWader |
| @protobufjs/aspromise | 1.1.2 | BSD-3-Clause | Copyright (c) 2016, Daniel Wirtz  All rights reserved. |
| @protobufjs/base64 | 1.1.2 | BSD-3-Clause | Copyright (c) 2016, Daniel Wirtz  All rights reserved. |
| @protobufjs/codegen | 2.0.5 | BSD-3-Clause | Copyright (c) 2016, Daniel Wirtz  All rights reserved. |
| @protobufjs/eventemitter | 1.1.1 | BSD-3-Clause | Copyright (c) 2016, Daniel Wirtz  All rights reserved. |
| @protobufjs/fetch | 1.1.1 | BSD-3-Clause | Copyright (c) 2016, Daniel Wirtz  All rights reserved. |
| @protobufjs/path | 1.1.2 | BSD-3-Clause | Copyright (c) 2016, Daniel Wirtz  All rights reserved. |
| @protobufjs/pool | 1.1.0 | BSD-3-Clause | Copyright (c) 2016, Daniel Wirtz  All rights reserved. |
| @protobufjs/utf8 | 1.1.2 | BSD-3-Clause | Copyright (c) 2016, Daniel Wirtz  All rights reserved. |
| @sagold/json-pointer | 5.1.2 | MIT | Copyright 2022 Sascha Goldhofer |
| @sagold/json-query | 6.2.0 | MIT | Copyright 2022 Sascha Goldhofer |
| @sideway/address | 4.1.5 | BSD-3-Clause | Copyright (c) 2019-2020, Sideway, Inc. and Project contributors |
| @sideway/pinpoint | 2.0.0 | BSD-3-Clause | Copyright (c) 2019-2020, Sideway. Inc, and project contributors |
| @sigstore/core | 3.2.0 | Apache-2.0 | - |
| @sigstore/protobuf-specs | 0.5.1 | Apache-2.0 | - |
| @sigstore/sign | 4.1.1 | Apache-2.0 | - |
| @sigstore/verify | 3.1.0 | Apache-2.0 | - |
| @statsig/client-core | 3.33.3 | ISC | - |
| @statsig/js-client | 3.33.3 | ISC | Copyright (c) 2021, Statsig, Inc. |
| @tufjs/canonical-json | 2.0.0 | MIT | Copyright (c) 2022 GitHub and the TUF Contributors |
| @typespec/ts-http-runtime | 0.3.6 | MIT | Copyright (c) Microsoft Corporation. |
| @vscode/fs-copyfile | 2.0.0 | MIT | - |
| @vscode/ripgrep | 1.15.9 | MIT | Copyright (c) Microsoft Corporation |
| @vscode/sandbox-runtime | 0.0.1 | Apache-2.0 | - |
| @webviewjs/webview | 0.4.0 | MIT | Copyright (c) 2024 Twilight |
| abbrev | 4.0.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| abstract-level | 3.1.1 | MIT | Copyright © 2013 Rod Vagg and the contributors to abstract-level. |
| acorn | 8.15.0 | MIT | Copyright (C) 2012-2022 by various contributors (see AUTHORS) |
| agent-base | 7.1.4 | MIT | Copyright (c) 2013 Nathan Rajlich |
| ansi-align | 3.0.1 | ISC | Copyright (c) 2016, Contributors |
| ansi-escapes | 4.3.2 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| ansi-regex | 6.2.2 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| ansi-styles | 6.2.3 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| anymatch | 3.1.3 | ISC | Copyright (c) 2019 Elan Shanker, Paul Miller (https://paulmillr.com) |
| application-config-path | 0.1.1 | MIT | Copyright (c) 2015 Linus Unnebäck |
| aproba | 2.1.0 | ISC | Copyright (c) 2015, Rebecca Turner |
| asynckit | 0.4.0 | MIT | Copyright (c) 2016 Alex Indigo |
| atomically | 2.1.1 | MIT | Copyright (c) 2020-present Fabio Spampinato |
| axios | 1.17.0 | MIT | - |
| azure-functions-nodejs-worker | 3.14.1 | (MIT OR Apache-2.0) | Copyright (c) .NET Foundation. All rights reserved. |
| babel-plugin-polyfill-regenerator | 0.5.4 | MIT | Copyright (c) 2014-present Nicolò Ribaudo and other contributors |
| balanced-match | 1.0.2 | MIT | Copyright (c) 2013 Julian Gruber &lt;&gt; |
| base64-js | 1.5.1 | MIT | Copyright (c) 2014 Jameson Little |
| bin-links | 6.0.0 | ISC | Copyright (c) npm, Inc. |
| bl | 4.1.0 | MIT | Copyright (c) 2013-2019 bl contributors |
| boxen | 8.0.1 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| brace-expansion | 5.0.5 | MIT | Copyright Julian Gruber |
| braces | 3.0.3 | MIT | Copyright (c) 2014-present, Jon Schlinkert. |
| buffer | 5.7.1 | MIT | Copyright (c) Feross Aboukhadijeh, and other contributors. |
| buffer-equal-constant-time | 1.0.1 | BSD-3-Clause | Copyright (c) 2013, GoInstant Inc., a salesforce.com company |
| buffer-from | 1.1.2 | MIT | Copyright (c) 2016, 2018 Linus Unnebäck |
| bufferutil | 4.0.1 | MIT | Copyright (c) 2011 Einar Otto Stangvik (http://2x.io) |
| call-bind-apply-helpers | 1.0.2 | MIT | Copyright (c) 2024 Jordan Harband |
| camelcase | 8.0.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| chalk | 5.6.2 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| charenc | 0.0.2 | BSD-3-Clause | - |
| chokidar | 3.5.3 | MIT | Copyright (c) 2012-2019 Paul Miller (https://paulmillr.com), Elan Shanker |
| chownr | 1.1.4 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| chrome-devtools-mcp | 1.1.0 | Apache-2.0 | - |
| ci-info | 4.4.0 | MIT | Copyright (c) 2016 Thomas Watson Steen |
| classic-level | 3.0.0 | MIT | Copyright (c) 2012 Rod Vagg and the contributors to classic-level and leveldown. |
| clean-css | 5.3.3 | MIT | Copyright (C) 2017 JakubPawlowicz.com |
| cli-cursor | 3.1.0 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| cli-spinners | 2.9.2 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| cliui | 8.0.1 | ISC | Copyright (c) 2015, Contributors |
| clone | 2.1.2 | MIT | Copyright © 2011-2015 Paul Vorbach |
| clone-buffer | 1.0.0 | MIT | Copyright (c) 2016 Blaine Bublitz, Eric Schoffstall and other contributors |
| clone-stats | 1.0.0 | MIT | Copyright (c) 2014 Hugh Kennedy |
| cloneable-readable | 1.1.3 | MIT | Copyright (c) 2016 Matteo Collina |
| color-convert | 2.0.1 | MIT | Copyright (c) 2011-2016 Heather Arthur |
| color-name | 1.1.4 | MIT | Copyright (c) 2015 Dmitry Ivanov |
| combined-stream | 1.0.8 | MIT | Copyright (c) 2011 Debuggable Limited |
| command-exists | 1.2.9 | MIT | Copyright (c) 2014 Matthew Conlen |
| concurrently | 7.6.0 | MIT | Copyright (c) 2015 Kimmo Brunfeldt |
| config-chain | 1.1.13 | MIT | Copyright (c) 2011 Dominic Tarr |
| configstore | 7.1.0 | BSD-2-Clause | Copyright (c) Google |
| continue | 2.0.0 | Apache-2.0 | Copyright 2023 Continue |
| cookie | 0.7.2 | MIT | Copyright (c) 2012-2014 Roman Shtylman |
| core-util-is | 1.0.3 | MIT | Copyright Node.js contributors. All rights reserved. |
| cross-spawn | 7.0.6 | MIT | Copyright (c) 2018 Made With MOXY Lda |
| crypt | 0.0.2 | BSD-3-Clause | - |
| cssesc | 3.0.0 | MIT | Copyright Mathias Bynens <https://mathiasbynens.be/> |
| decompress-response | 6.0.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| deep-extend | 0.6.0 | MIT | Copyright (c) 2013-2018, Viacheslav Lotsmanov |
| deepmerge | 4.3.1 | MIT | Copyright (c) 2012 James Halliday, Josh Duff, and other contributors |
| default-browser | 5.5.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| default-browser-id | 5.0.1 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| default-gateway | 6.0.3 | BSD-2-Clause | Copyright (c) silverwind |
| defaults | 1.0.4 | MIT | Copyright (c) 2022 Sindre Sorhus |
| define-lazy-prop | 3.0.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| delayed-stream | 1.0.0 | MIT | Copyright (c) 2011 Debuggable Limited |
| depd | 2.0.0 | MIT | Copyright (c) 2014-2018 Douglas Christopher Wilson |
| destroy | 1.2.0 | MIT | Copyright (c) 2014 Jonathan Ong |
| detect-libc | 2.1.2 | Apache-2.0 | - |
| devcert | 1.2.3 | MIT | - |
| discontinuous-range | 1.0.0 | MIT | Copyright (c) 2014 David Tudury |
| dot-case | 3.0.4 | MIT | Copyright (c) 2014 Blake Embrey |
| dot-prop | 9.0.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| dotenv | 16.6.1 | BSD-2-Clause | Copyright (c) 2015, Scott Motte |
| dunder-proto | 1.0.1 | MIT | Copyright (c) 2024 ECMAScript Shims |
| eastasianwidth | 0.2.0 | MIT | - |
| ebnf | 1.9.1 | MIT | Copyright (c) 2016 Agustin Mendez |
| ee-first | 1.1.1 | MIT | Copyright (c) 2014 Jonathan Ong |
| electron-to-chromium | 1.5.250 | ISC | Copyright 2018 Kilian Valkhof |
| encodeurl | 2.0.0 | MIT | Copyright (c) 2016 Douglas Christopher Wilson |
| entities | 4.5.0 | BSD-2-Clause | Copyright (c) Felix Böhm |
| env-paths | 2.2.1 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| eol | 0.9.1 | MIT | - |
| es-define-property | 1.0.1 | MIT | Copyright (c) 2024 Jordan Harband |
| es-errors | 1.3.0 | MIT | Copyright (c) 2024 Jordan Harband |
| es-object-atoms | 1.1.2 | MIT | Copyright (c) 2024 Jordan Harband |
| es-set-tostringtag | 2.1.0 | MIT | Copyright (c) 2022 ECMAScript Shims |
| es6-iterator | 2.0.3 | MIT | Copyright (C) 2013-2017 Mariusz Nowak (www.medikoo.com) |
| es6-symbol | 3.1.3 | ISC | Copyright (c) 2013-2019, Mariusz Nowak, @medikoo, medikoo.com |
| escalade | 3.2.0 | MIT | Copyright (c) Luke Edwards (lukeed.com) |
| escape-goat | 4.0.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| escape-html | 1.0.3 | MIT | Copyright (c) 2012-2013 TJ Holowaychuk |
| esutils | 2.0.3 | BSD-2-Clause | - |
| etag | 1.8.1 | MIT | Copyright (c) 2014-2016 Douglas Christopher Wilson |
| eventemitter3 | 4.0.7 | MIT | Copyright (c) 2014 Arnout Kazemier |
| execa | 5.1.1 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| expand-template | 2.0.3 | (MIT OR WTFPL) | Copyright (c) 2018 Lars-Magnus Skog |
| ext | 1.4.0 | ISC | Copyright (c) 2011-2019, Mariusz Nowak, @medikoo, medikoo.com |
| fast-copy | 3.0.2 | MIT | Copyright (c) 2018 Tony Quetano |
| fastest-levenshtein | 1.0.16 | MIT | Copyright (c) 2020 Kasper Unn Weihe |
| fdir | 6.5.0 | MIT | Copyright 2023 Abdullah Atta |
| file-uri-to-path | 1.0.0 | MIT | Copyright (c) 2014 Nathan Rajlich |
| fill-range | 7.1.1 | MIT | Copyright (c) 2014-present, Jon Schlinkert. |
| finalhandler | 1.3.2 | MIT | Copyright (c) 2014-2022 Douglas Christopher Wilson |
| follow-redirects | 1.16.0 | MIT | Copyright 2014–present Olivier Lalonde, James Talmage, Ruben Verborgh |
| foreground-child | 3.3.1 | ISC | Copyright (c) 2015-2023 Isaac Z. Schlueter and Contributors |
| form-data | 4.0.5 | MIT | Copyright (c) 2012 Felix Geisendörfer and contributors |
| foundry-local-sdk | 1.2.3 | MIT | - |
| fresh | 0.5.2 | MIT | Copyright (c) 2012 TJ Holowaychuk |
| fs-minipass | 3.0.3 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| fs-readdir-recursive | 1.1.0 | MIT | Copyright (c) 2014 Jonathan Ong |
| fs.realpath | 1.0.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| function-bind | 1.1.2 | MIT | Copyright (c) 2013 Raynos. |
| gensync | 1.0.0-beta.2 | MIT | Copyright 2018 Logan Smyth |
| get-caller-file | 2.0.5 | ISC | Copyright 2018 Stefan Penner |
| get-east-asian-width | 1.6.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| get-intrinsic | 1.3.0 | MIT | Copyright (c) 2020 Jordan Harband |
| get-port | 5.1.1 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| get-proto | 1.0.1 | MIT | Copyright (c) 2025 Jordan Harband |
| get-stream | 6.0.1 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| github-from-package | 0.0.0 | MIT | - |
| glob | 10.5.0 | ISC | Copyright (c) 2009-2023 Isaac Z. Schlueter and Contributors |
| glob-parent | 5.1.2 | ISC | Copyright (c) 2015, 2019 Elan Shanker |
| global-directory | 4.0.1 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| gopd | 1.2.0 | MIT | Copyright (c) 2022 Jordan Harband |
| graceful-fs | 4.2.11 | ISC | Copyright (c) 2011-2022 Isaac Z. Schlueter, Ben Noordhuis, and Contributors |
| handlebars | 4.7.9 | MIT | Copyright (C) 2011-2019 by Yehuda Katz |
| has-symbols | 1.1.0 | MIT | Copyright (c) 2016 Jordan Harband |
| has-tostringtag | 1.0.2 | MIT | Copyright (c) 2021 Inspect JS |
| hosted-git-info | 9.0.2 | ISC | Copyright (c) 2015, Rebecca Turner |
| html-minifier-terser | 7.2.0 | MIT | Copyright (c) 2010-2019 Juriy "kangax" Zaytsev |
| http-cache-semantics | 4.2.0 | BSD-2-Clause | Copyright 2016-2018 Kornel Lesiński |
| http-errors | 2.0.1 | MIT | Copyright (c) 2014 Jonathan Ong |
| http-proxy-agent | 7.0.2 | MIT | Copyright (c) 2013 Nathan Rajlich |
| human-signals | 2.1.0 | Apache-2.0 | - |
| iconv-lite | 0.7.2 | MIT | Copyright (c) 2011 Alexander Shtuchkin |
| ieee754 | 1.2.1 | BSD-3-Clause | Copyright 2008 Fair Oaks Labs, Inc. |
| ignore-walk | 8.0.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| inflight | 1.0.6 | ISC | Copyright (c) Isaac Z. Schlueter |
| inherits | 2.0.4 | ISC | Copyright (c) Isaac Z. Schlueter |
| ini | 6.0.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| init-package-json | 8.2.5 | ISC | Copyright npm, Inc. |
| internal-ip | 6.2.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| ip-address | 10.1.0 | MIT | Copyright (C) 2011 by Beau Gunderson |
| ipaddr.js | 1.9.1 | MIT | Copyright (C) 2011-2017 whitequark |
| is-buffer | 1.1.6 | MIT | Copyright (c) Feross Aboukhadijeh |
| is-core-module | 2.13.1 | MIT | Copyright (c) 2014 Dave Justice |
| is-docker | 3.0.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| is-extglob | 2.1.1 | MIT | Copyright (c) 2014-2016, Jon Schlinkert |
| is-fullwidth-code-point | 3.0.0 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| is-glob | 4.0.3 | MIT | Copyright (c) 2014-2017, Jon Schlinkert. |
| is-inside-container | 1.0.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| is-installed-globally | 1.0.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| is-number | 7.0.0 | MIT | Copyright (c) 2014-present, Jon Schlinkert. |
| is-stream | 2.0.1 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| is-typedarray | 1.0.0 | MIT | - |
| is-valid-domain | 0.1.6 | MIT | Copyright (C) 2014 Miguel Mota |
| is-wsl | 3.1.1 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| isarray | 1.0.0 | MIT | - |
| isexe | 2.0.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| joi | 17.13.4 | BSD-3-Clause | Copyright (c) 2012-2022, Project contributors. |
| jose | 6.2.3 | MIT | Copyright (c) 2018 Filip Skokan |
| jpeg-js | 0.4.4 | BSD-3-Clause | Copyright (c) 2014, Eugene Ware |
| jsesc | 2.5.2 | MIT | Copyright Mathias Bynens <https://mathiasbynens.be/> |
| json-parse-even-better-errors | 5.0.0 | MIT | Copyright 2017 Kat Marchán |
| json-schema-library | 9.3.5 | MIT | Copyright 2022 Sascha Goldhofer |
| json-stringify-nice | 1.1.4 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| json5 | 2.2.3 | MIT | Copyright (c) 2012-2018 Aseem Kishore, and [others]. |
| jsonparse | 1.3.1 | MIT | Copyright (c) 2012 Tim Caswell |
| jsonschema | 1.5.0 | MIT | Copyright (C) 2012-2015 Tom de Grunt |
| just-diff | 6.0.2 | MIT | Copyright (c) 2016 angus croll |
| just-diff-apply | 5.5.0 | MIT | Copyright (c) 2016 angus croll |
| jwa | 2.0.1 | MIT | Copyright (c) 2013 Brian J. Brennan |
| jws | 4.0.1 | MIT | Copyright (c) 2013 Brian J. Brennan |
| jwt-decode | 4.0.0 | MIT | Copyright (c) 2015 Auth0, Inc. (http://auth0.com) |
| keytar | 7.9.0 | MIT | Copyright (c) 2013 GitHub Inc. |
| kleur | 3.0.3 | MIT | Copyright (c) Luke Edwards (lukeed.com) |
| ky | 1.14.3 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| level-supports | 6.2.0 | MIT | Copyright (c) 2019 The contributors to level-supports. |
| level-transcoder | 1.0.1 | MIT | Copyright © 2012 The contributors to level-transcoder and level-codec. |
| libnpmaccess | 10.0.3 | ISC | Copyright npm, Inc |
| libnpmdiff | 8.1.6 | ISC | Copyright (c) GitHub Inc. |
| libnpmfund | 7.0.20 | ISC | Copyright (c) npm Inc. |
| libnpmorg | 8.0.1 | ISC | Copyright npm, Inc |
| libnpmpack | 9.1.6 | ISC | Copyright npm, Inc |
| libnpmpublish | 11.1.3 | ISC | Copyright npm, Inc |
| libnpmsearch | 9.0.1 | ISC | Copyright npm, Inc |
| libnpmteam | 8.0.2 | ISC | Copyright npm, Inc |
| libnpmversion | 8.0.3 | ISC | Copyright (c) Isaac Z. Schlueter |
| lodash | 4.18.1 | MIT | Copyright OpenJS Foundation and other contributors <https://openjsf.org/> |
| lodash.camelcase | 4.3.0 | MIT | Copyright jQuery Foundation and other contributors <https://jquery.org/> |
| lodash.debounce | 4.0.8 | MIT | Copyright jQuery Foundation and other contributors <https://jquery.org/> |
| lodash.includes | 4.3.0 | MIT | Copyright jQuery Foundation and other contributors <https://jquery.org/> |
| lodash.isboolean | 3.0.3 | MIT | Copyright 2012-2016 The Dojo Foundation <http://dojofoundation.org/> |
| lodash.isinteger | 4.0.4 | MIT | Copyright jQuery Foundation and other contributors <https://jquery.org/> |
| lodash.isnumber | 3.0.3 | MIT | Copyright 2012-2016 The Dojo Foundation <http://dojofoundation.org/> |
| lodash.isplainobject | 4.0.6 | MIT | Copyright jQuery Foundation and other contributors <https://jquery.org/> |
| lodash.isstring | 4.0.1 | MIT | Copyright 2012-2016 The Dojo Foundation <http://dojofoundation.org/> |
| lodash.once | 4.1.1 | MIT | Copyright jQuery Foundation and other contributors <https://jquery.org/> |
| log-symbols | 4.1.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| long | 5.3.2 | Apache-2.0 | - |
| lower-case | 2.0.2 | MIT | Copyright (c) 2014 Blake Embrey |
| lru-cache | 10.4.3 | ISC | Copyright (c) 2010-2023 Isaac Z. Schlueter and Contributors |
| make-dir | 2.1.0 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| make-fetch-happen | 15.0.5 | ISC | Copyright 2017-2022 (c) npm, Inc. |
| math-intrinsics | 1.1.0 | MIT | Copyright (c) 2024 ECMAScript Shims |
| maybe-combine-errors | 1.0.0 | MIT | - |
| md5 | 2.3.0 | BSD-3-Clause | Copyright © 2011-2012, Paul Vorbach. |
| merge-stream | 2.0.0 | MIT | Copyright (c) Stephen Sugden (stephensugden.com) |
| mime | 1.6.0 | MIT | Copyright (c) 2010 Benjamin Thomas, Robert Kieffer |
| mime-types | 2.1.35 | MIT | Copyright (c) 2014 Jonathan Ong |
| mimic-response | 3.1.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| minipass | 3.3.6 | ISC | Copyright (c) 2017-2022 npm, Inc., Isaac Z. Schlueter, and Contributors |
| minipass-collect | 2.0.1 | ISC | Copyright (c) 2019-2023 Isaac Z. Schlueter and Contributors |
| minipass-pipeline | 1.2.4 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| minipass-sized | 2.0.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| minizlib | 3.1.0 | MIT | Copyright (c) 2017-2023 Isaac Z. Schlueter and Contributors |
| mkdirp | 0.5.6 | MIT | Copyright 2010 James Halliday |
| module-error | 1.0.2 | MIT | Copyright (c) Vincent Weevers |
| moo | 0.5.3 | BSD-3-Clause | Copyright (c) 2017, Tim Radvan (tjvr) |
| ms | 2.1.3 | MIT | Copyright (c) 2020 Vercel, Inc. |
| mute-stream | 3.0.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| napi-build-utils | 2.0.0 | MIT | Copyright (c) 2018 inspiredware |
| negotiator | 1.0.0 | MIT | Copyright (c) 2012-2014 Federico Romero |
| next-tick | 1.0.0 | MIT | Copyright (C) 2012-2016 Mariusz Nowak |
| no-case | 3.0.4 | MIT | Copyright (c) 2014 Blake Embrey |
| node-abi | 3.92.0 | MIT | Copyright (c) 2016 Lukas Geiger |
| node-addon-api | 5.1.0 | MIT | Copyright (c) 2017 Node.js API collaborators |
| node-fetch | 2.7.0 | MIT | Copyright (c) 2016 David Frank |
| node-gyp | 12.3.0 | MIT | Copyright (c) 2012 Nathan Rajlich |
| node-pty | 1.1.0-beta39 | MIT | Copyright (c) 2012-2015, Christopher Jeffrey (https://github.com/chjj/) |
| nopt | 9.0.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| normalize-path | 3.0.0 | MIT | Copyright (c) 2014-2018, Jon Schlinkert. |
| npm-audit-report | 7.0.0 | ISC | Copyright (c) npm, Inc. |
| npm-bundled | 5.0.0 | ISC | Copyright (c) npm, Inc. and Contributors |
| npm-install-checks | 8.0.0 | BSD-2-Clause | Copyright (c) Robert Kowalski and Isaac Z. Schlueter ("Authors") |
| npm-normalize-package-bin | 5.0.0 | ISC | Copyright (c) npm, Inc. |
| npm-package-arg | 13.0.2 | ISC | Copyright (c) npm, Inc. |
| npm-packlist | 10.0.4 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| npm-pick-manifest | 11.0.3 | ISC | Copyright (c) npm, Inc. |
| npm-profile | 12.0.1 | ISC | Copyright npm, Inc. |
| npm-registry-fetch | 19.1.1 | ISC | Copyright npm, Inc. |
| npm-run-path | 4.0.1 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| npm-user-validate | 4.0.0 | BSD-2-Clause | Copyright (c) Robert Kowalski |
| ollama-ai-agent | 1.3.5 | MIT | Copyright (c) 2025 Nishant Unavane |
| on-finished | 2.4.1 | MIT | Copyright (c) 2013 Jonathan Ong |
| once | 1.4.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| onetime | 5.1.2 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| open | 10.2.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| openid-client | 6.8.4 | MIT | Copyright (c) 2016 Filip Skokan |
| ora | 5.4.1 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| os-tmpdir | 1.0.2 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| p-event | 4.2.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| p-finally | 1.0.0 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| p-map | 7.0.4 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| p-timeout | 3.2.0 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| package-json | 10.0.1 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| param-case | 3.0.4 | MIT | Copyright (c) 2014 Blake Embrey |
| parse-conflict-json | 5.0.1 | ISC | Copyright npm, Inc. |
| parseurl | 1.3.3 | MIT | Copyright (c) 2014 Jonathan Ong |
| pascal-case | 3.1.2 | MIT | Copyright (c) 2014 Blake Embrey |
| path-is-absolute | 1.0.1 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| path-key | 3.1.1 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| path-parse | 1.0.7 | MIT | Copyright (c) 2015 Javier Blanco |
| pem | 1.14.8 | MIT | Copyright (c) 2012 Andris Reinman |
| picocolors | 1.1.1 | ISC | Copyright (c) 2021-2024 Oleksii Raspopov, Kostiantyn Denysov, Anton Verinov |
| picomatch | 4.0.4 | MIT | Copyright (c) 2017-present, Jon Schlinkert. |
| pify | 4.0.1 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| pixelmatch | 7.1.0 | ISC | Copyright (c) 2025, Mapbox |
| playwright | 1.57.0 | Apache-2.0 | - |
| playwright-core | 1.57.0 | Apache-2.0 | - |
| pngjs | 7.0.0 | MIT | - |
| prebuild-install | 7.1.3 | MIT | Copyright (c) 2015 Mathias Buus |
| proc-log | 6.1.0 | ISC | Copyright (c) GitHub, Inc. |
| process-nextick-args | 2.0.1 | MIT | - |
| proggy | 4.0.0 | ISC | Copyright (c) GitHub, Inc. |
| promise-all-reject-late | 1.0.1 | ISC | Copyright (c) Isaac Z. Schlueter |
| promise-call-limit | 3.0.2 | ISC | Copyright (c) Isaac Z. Schlueter |
| prompts | 2.4.2 | MIT | Copyright (c) 2018 Terkel Gjervig Nielsen |
| promzard | 3.0.1 | ISC | Copyright (c) Isaac Z. Schlueter |
| proto-list | 1.2.4 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| protobufjs | 7.6.6 | BSD-3-Clause | Copyright (c) 2016, Daniel Wirtz  All rights reserved. |
| proxy-from-env | 2.1.0 | MIT | Copyright (C) 2016-2018 Rob Wu |
| punycode | 2.3.1 | MIT | Copyright Mathias Bynens <https://mathiasbynens.be/> |
| pupa | 3.3.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| python | 2026.4.0 | MIT | Copyright (c) Microsoft Corporation. All rights reserved. |
| qrcode-terminal | 0.12.0 | Apache 2.0 | - |
| randexp | 0.4.6 | MIT | Copyright (C) 2011 by Roly Fentanes |
| range-parser | 1.2.1 | MIT | Copyright (c) 2012-2014 TJ Holowaychuk |
| rc | 1.2.8 | (BSD-2-Clause OR MIT OR Apache-2.0) | Copyright (c) 2011 Dominic Tarr |
| read | 5.0.1 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| read-cmd-shim | 6.0.0 | ISC | Copyright (c) 2015, Rebecca Turner |
| readdirp | 3.6.0 | MIT | Copyright (c) 2012-2019 Thorsten Lorenz, Paul Miller (https://paulmillr.com) |
| regenerate | 1.4.2 | MIT | Copyright Mathias Bynens <https://mathiasbynens.be/> |
| regenerate-unicode-properties | 10.1.1 | MIT | Copyright Mathias Bynens <https://mathiasbynens.be/> |
| regenerator-transform | 0.15.2 | MIT | Copyright (c) 2014-present, Facebook, Inc. |
| regexpu-core | 5.3.2 | MIT | Copyright Mathias Bynens <https://mathiasbynens.be/> |
| registry-auth-token | 5.1.1 | MIT | Copyright (c) 2016 Espen Hovlandsdal |
| relateurl | 0.2.7 | MIT | Copyright (c) Steven Vachon (svachon.com) |
| remove-trailing-separator | 1.1.0 | ISC | - |
| replace-ext | 1.0.1 | MIT | Copyright (c) 2014 Blaine Bublitz, Eric Schoffstall and other contributors |
| require-directory | 2.1.1 | MIT | Copyright (c) 2011 Troy Goode |
| resolve | 1.22.8 | MIT | Copyright (c) 2012 James Halliday |
| ret | 0.1.15 | MIT | Copyright (C) 2011 by Roly Fentanes |
| rimraf | 5.0.10 | ISC | Copyright (c) 2011-2023 Isaac Z. Schlueter and Contributors |
| ruby-lsp | 0.10.3 | MIT | Copyright (c) 2022-present, Shopify Inc. |
| run-applescript | 7.1.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| safe-buffer | 5.2.1 | MIT | Copyright (c) Feross Aboukhadijeh |
| safer-buffer | 2.1.2 | MIT | Copyright (c) 2018 Nikita Skovoroda |
| semver | 7.8.5 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| send | 0.19.2 | MIT | Copyright (c) 2012 TJ Holowaychuk |
| serve-static | 1.16.3 | MIT | Copyright (c) 2010 Sencha Inc. |
| setprototypeof | 1.2.0 | ISC | Copyright (c) 2015, Wes Todd |
| sharp | 0.35.4 | Apache-2.0 | - |
| shebang-command | 2.0.0 | MIT | Copyright (c) Kevin Mårtensson (github.com/kevva) |
| signal-exit | 4.1.0 | ISC | Copyright (c) 2015-2023 Benjamin Coe, Isaac Z. Schlueter, and Contributors |
| sigstore | 4.1.0 | Apache-2.0 | - |
| simple-concat | 1.0.1 | MIT | Copyright (c) Feross Aboukhadijeh |
| simple-get | 4.0.1 | MIT | Copyright (c) Feross Aboukhadijeh |
| sisteransi | 1.0.5 | MIT | Copyright (c) 2018 Terkel Gjervig Nielsen |
| smart-buffer | 4.2.0 | MIT | Copyright (c) 2013-2017 Josh Glazebrook |
| smtp-address-parser | 1.0.10 | MIT | Copyright (c) 2021 Gene Hightower |
| socks | 2.8.7 | MIT | Copyright (c) 2013 Josh Glazebrook |
| socks-proxy-agent | 8.0.5 | MIT | Copyright (c) 2013 Nathan Rajlich |
| source-map | 0.6.1 | BSD-3-Clause | Copyright (c) 2009-2011, Mozilla Foundation and contributors |
| spawn-command | 0.0.2 | MIT | Copyright (C) 2011 by Maciej Małecki |
| spdx-expression-parse | 4.0.0 | MIT | Copyright (c) 2015 Kyle E. Mitchell & other authors listed in AUTHORS |
| ssri | 13.0.1 | ISC | Copyright 2021 (c) npm, Inc. |
| statuses | 2.0.2 | MIT | Copyright (c) 2014 Jonathan Ong |
| steamworks.js | 0.4.0 | MIT | Copyright (c) 2022 Gabriel Francisco Dos Santos |
| string-width | 7.2.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| string_decoder | 1.3.0 | MIT | Copyright Node.js contributors. All rights reserved. |
| strip-ansi | 7.2.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| strip-final-newline | 2.0.0 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| strip-json-comments | 2.0.1 | MIT | Copyright (c) Sindre Sorhus (sindresorhus.com) |
| stubborn-fs | 2.0.0 | MIT | Copyright (c) 2022-present Fabio Spampinato |
| stubborn-utils | 1.0.2 | MIT | Copyright (c) 2025-present Fabio Spampinato |
| sudo-prompt | 8.2.5 | MIT | Copyright (c) 2015 Joran Dirk Greef |
| supports-color | 10.2.2 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| supports-preserve-symlinks-flag | 1.0.0 | MIT | Copyright (c) 2022 Inspect JS |
| tar-fs | 2.1.4 | MIT | Copyright (c) 2014 Mathias Buus |
| terser | 5.44.1 | BSD-2-Clause | Copyright 2012-2018 (c) Mihai Bazon |
| tiny-relative-date | 2.0.2 | MIT | Copyright (c) 2017 Joseph Wynn |
| tinyglobby | 0.2.16 | MIT | Copyright (c) 2024 Madeline Gurriarán |
| tmp | 0.0.33 | MIT | Copyright (c) 2014 KARASZI István |
| to-regex-range | 5.0.1 | MIT | Copyright (c) 2015-present, Jon Schlinkert. |
| toidentifier | 1.0.1 | MIT | Copyright (c) 2016 Douglas Christopher Wilson |
| tree-kill | 1.2.2 | MIT | Copyright (c) 2018 Peter Krumins |
| treeverse | 3.0.0 | ISC | Copyright (c) npm, Inc. and Contributors |
| tuf-js | 4.1.0 | MIT | Copyright (c) 2022 GitHub and the TUF Contributors |
| tunnel-agent | 0.6.0 | Apache-2.0 | - |
| typedarray-to-buffer | 3.1.5 | MIT | Copyright (c) Feross Aboukhadijeh |
| typescript | 6.0.3 | Apache-2.0 | - |
| underscore | 1.13.8 | MIT | Copyright (c) 2009-2022 Jeremy Ashkenas, Julian Gonggrijp, and DocumentCloud and Investigative Reporters & Editors |
| unicode-canonical-property-names-ecmascript | 2.0.0 | MIT | Copyright Mathias Bynens <https://mathiasbynens.be/> |
| unicode-match-property-ecmascript | 2.0.0 | MIT | Copyright Mathias Bynens <https://mathiasbynens.be/> |
| unicode-match-property-value-ecmascript | 2.1.0 | MIT | Copyright Mathias Bynens <https://mathiasbynens.be/> |
| unpipe | 1.0.0 | MIT | Copyright (c) 2015 Douglas Christopher Wilson |
| update-browserslist-db | 1.1.4 | MIT | Copyright 2022 Andrey Sitnik and other contributors |
| update-notifier | 7.3.1 | BSD-2-Clause | Copyright Google |
| util-deprecate | 1.0.2 | MIT | Copyright (c) 2014 Nathan Rajlich |
| uuid | 11.1.1 | MIT | Copyright (c) 2010-2020 Robert Kieffer and other contributors |
| validate-npm-package-name | 7.0.2 | ISC | Copyright (c) 2015, npm, Inc |
| vinyl | 2.2.1 | MIT | Copyright (c) 2013 Blaine Bublitz, Eric Schoffstall and other contributors |
| vinyl-sourcemaps-apply | 0.2.1 | ISC | - |
| vscode-python-envs | 1.20.1 | MIT | Copyright (c) Microsoft Corporation. |
| wait-on | 7.2.0 | MIT | Copyright (c) 2015 Jeff Barczewski |
| walk-up-path | 4.0.0 | ISC | Copyright (c) 2020-2023 Isaac Z. Schlueter |
| wcwidth | 1.0.1 | MIT | Copyright (C) 2012 by Jun Woong. |
| webidl-conversions | 3.0.1 | BSD-2-Clause | Copyright (c) 2014, Domenic Denicola |
| websocket | 1.0.32 | Apache-2.0 | - |
| whatwg-url | 5.0.0 | MIT | Copyright (c) 2015–2016 Sebastian Mayr |
| which | 6.0.1 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| windows-notification-state | 2.0.0 | MIT | Copyright (c) 2017 Felix Rieseberg |
| wordwrap | 1.0.0 | MIT | - |
| workerpool | 9.3.3 | Apache-2.0 | - |
| wrap-ansi | 9.0.2 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| wrappy | 1.0.2 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| write-file-atomic | 7.0.1 | ISC | Copyright (c) 2015, Rebecca Turner |
| ws | 8.18.3 | MIT | Copyright (c) 2011 Einar Otto Stangvik |
| wsl-utils | 0.1.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| xdg-basedir | 5.1.0 | MIT | Copyright (c) Sindre Sorhus (https://sindresorhus.com) |
| y18n | 5.0.8 | ISC | Copyright (c) 2015, Contributors |
| yaeti | 0.0.6 | MIT | Copyright (c) 2015 Iñaki Baz Castillo, |
| yallist | 4.0.0 | ISC | Copyright (c) Isaac Z. Schlueter and Contributors |
| yargs-parser | 21.1.1 | ISC | Copyright (c) 2016, Contributors |
| zod | 4.4.3 | MIT | Copyright (c) 2025 Colin McDonnell |
