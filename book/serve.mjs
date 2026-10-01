import http from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.map': 'application/json', '.ipynb': 'application/json',
  '.webmanifest': 'application/manifest+json', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon',
  '.woff': 'font/woff', '.woff2': 'font/woff2', '.ttf': 'font/ttf', '.eot': 'application/vnd.ms-fontobject', '.wasm': 'application/wasm', '.pdf': 'application/pdf',
  '.whl': 'application/zip', '.zip': 'application/zip', '.adoc': 'text/plain; charset=utf-8', '.py': 'text/plain; charset=utf-8', '.txt': 'text/plain; charset=utf-8',
};

/** Serves a static directory under a base path, e.g. dist/ at /app/refresh/ like the parent site. */
export function serve(directory, { port = 0, host = '127.0.0.1', base = '/' } = {}) {
  const root = path.resolve(directory);
  const server = http.createServer(async (request, response) => {
    const { pathname } = new URL(request.url, 'http://localhost');
    if (!pathname.startsWith(base)) {
      response.writeHead(302, { location: base }).end();
      return;
    }
    let filename = path.resolve(root, `.${decodeURIComponent(pathname.slice(base.length - 1))}`);
    if (filename !== root && !filename.startsWith(root + path.sep)) {
      response.writeHead(403).end();
      return;
    }
    try {
      if ((await stat(filename)).isDirectory()) filename = path.join(filename, 'index.html');
      await stat(filename);
      response.writeHead(200, { 'content-type': types[path.extname(filename)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
      createReadStream(filename).pipe(response);
    } catch {
      response.writeHead(404, { 'content-type': 'text/plain' }).end('Not found');
    }
  });
  return new Promise((resolve) => server.listen(port, host, () => resolve({ server, url: `http://${host === '0.0.0.0' ? 'localhost' : host}:${server.address().port}${base}` })));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const root = fileURLToPath(new URL('..', import.meta.url));
  const { url } = await serve(path.join(root, 'dist'), { port: Number(process.env.PORT ?? 1414), host: 'localhost', base: '/app/refresh/' });
  console.log(`Serving dist/ at ${url}`);
}
