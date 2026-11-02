const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const app = express();
const port = 3131;

// Connect to the SQLite database
const db = new sqlite3.Database('../blocks.db', (err) => {
    if (err) {
        console.error('Error connecting to the database:', err.message);
    } else {
        console.log('Connected to the blocks database.');
        checkLatestBlock(); // Call the new function when the database connection is established
    }
});

// New function to check and log the latest block
function checkLatestBlock() {
    db.get("SELECT MAX(block_id) as latest_block FROM blocks", (err, row) => {
        if (err) {
            console.error('Error fetching latest block:', err.message);
        } else {
            const latestBlock = row ? row.latest_block : 0;
            console.log(`Current latest block: ${latestBlock}`);
        }
    });
}

// Function to get the latest block and calculate countdown
function getLatestBlockAndCountdown() {
    return new Promise((resolve, reject) => {
        db.get("SELECT MAX(block_id) as latest_block FROM blocks", (err, row) => {
            if (err) {
                reject(err);
            } else {
                const latestBlock = row ? row.latest_block : 0;
                const targetBlock = 62500001;
                const countdown = Math.max(0, targetBlock - latestBlock);
                const halvingDate = new Date(Date.now() + countdown * 1000);
                resolve({ latestBlock, countdown, targetBlock, halvingDate });
            }
        });
    });
}

// Serve the HTML page
app.get('/countdown', (req, res) => {
    res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, minimal-ui">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <title>XENBLOCKs HALVING COUNTDOWN</title>
        <style>
            html, body {
                height: 100%;
                width: 100%;
                overflow: hidden;
            }
            body {
                background-color: #000;
                color: #0f0;
                font-family: 'Courier New', monospace;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                align-items: center;
                margin: 0;
                padding: 20px;
                box-sizing: border-box;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
            }
            header {
                width: 100%;
                text-align: center;
            }
            main {
                flex-grow: 1;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            footer {
                width: 100%;
                text-align: center;
            }
            h1 {
                font-size: 2.5em;
                text-shadow: 0 0 5px #0a0;
                margin-bottom: 20px;
                text-align: center;
                animation: glow 1.5s ease-in-out infinite alternate;
            }
            #countdown {
                font-size: 2.6em;
                text-shadow: 0 0 10px #0f0;
                text-align: center;
                margin-bottom: 20px;
            }
            #latest-block, #halving-block {
                font-size: 1.3em;
                text-shadow: 0 0 5px #0f0;
                text-align: center;
                margin: 10px 0;
            }
            #halving-date {
                font-size: 1.04em;
                text-shadow: 0 0 5px #0f0;
                text-align: center;
                margin-top: 20px;
            }
            .matrix-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: -1;
            }
            @keyframes glow {
                from {
                    text-shadow: 0 0 2px #0a0, 0 0 4px #0a0, 0 0 6px #0a0, 0 0 8px #0a0;
                }
                to {
                    text-shadow: 0 0 4px #0a0, 0 0 8px #0a0, 0 0 12px #0a0, 0 0 16px #0a0;
                }
            }
            @media (max-width: 767px) {
                #countdown, #latest-block, #halving-block, #halving-date {
                    font-weight: bold;
                }
                #halving-date {
                    font-size: 0.728em; /* 30% smaller than 1.04em */
                }
            }
            @media (min-width: 768px) {
                h1 {
                    font-size: 4em;
                }
                #countdown {
                    font-size: 4em;
                }
                #latest-block, #halving-block {
                    font-size: 2em;
                }
                #halving-date {
                    font-size: 1.6em;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>XENBLOCKs HALVING COUNTDOWN</h1>
        </header>
        <main>
            <div id="countdown">Loading...</div>
            <div id="latest-block">Latest Block: Loading...</div>
            <div id="halving-block">Halving Block: 62,500,001</div>
        </main>
        <footer>
            <div id="halving-date">Halving Date: Loading...</div>
        </footer>
        <canvas class="matrix-bg" id="matrix"></canvas>
        <script>
            function updateCountdown() {
                fetch('/latest-block')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('countdown').textContent = data.countdown.toLocaleString();
                        document.getElementById('latest-block').textContent = 'Latest Block: ' + data.latestBlock.toLocaleString();
                        document.getElementById('halving-date').textContent = 'Halving Date: ' + new Date(data.halvingDate).toUTCString();
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('countdown').textContent = 'Error loading countdown';
                        document.getElementById('latest-block').textContent = 'Error loading latest block';
                        document.getElementById('halving-date').textContent = 'Error loading halving date';
                    });
            }

            setInterval(updateCountdown, 10000);
            updateCountdown();

            // Matrix rain effect
            const canvas = document.getElementById('matrix');
            const ctx = canvas.getContext('2d');

            function resizeCanvas() {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            }

            window.addEventListener('resize', resizeCanvas);
            resizeCanvas();

            const chars = '0123456789ABCDEF';
            const charSize = 14;
            let columns = Math.floor(canvas.width / charSize);
            let drops = [];

            function initDrops() {
                columns = Math.floor(canvas.width / charSize);
                drops = [];
                for (let i = 0; i < columns; i++) {
                    drops[i] = 1;
                }
            }

            initDrops();
            window.addEventListener('resize', initDrops);

            function draw() {
                ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                ctx.fillStyle = '#0f0';
                ctx.font = charSize + 'px monospace';

                for (let i = 0; i < drops.length; i++) {
                    const text = chars[Math.floor(Math.random() * chars.length)];
                    ctx.fillText(text, i * charSize, drops[i] * charSize);

                    if (drops[i] * charSize > canvas.height && Math.random() > 0.975) {
                        drops[i] = 0;
                    }
                    drops[i]++;
                }
            }

            setInterval(draw, 33);

            // Hide address bar on mobile
            function hideAddressBar() {
                if(document.documentElement.scrollHeight < window.outerHeight / window.devicePixelRatio) {
                    document.documentElement.style.height = (window.outerHeight / window.devicePixelRatio) + 'px';
                }
                setTimeout(window.scrollTo(0, 1), 0);
            }

            window.addEventListener('load', hideAddressBar);
            window.addEventListener('orientationchange', hideAddressBar);

            // Add touch and scroll event listeners to minimize address bar
            window.addEventListener('touchstart', hideAddressBar);
            window.addEventListener('touchmove', hideAddressBar);
            window.addEventListener('scroll', hideAddressBar);
        </script>
    </body>
    </html>
    `);
});

// API endpoint to get the latest block and countdown
app.get('/latest-block', async (req, res) => {
    try {
        const { latestBlock, countdown, targetBlock, halvingDate } = await getLatestBlockAndCountdown();
        res.json({ latestBlock, countdown, targetBlock, halvingDate });
    } catch (error) {
        console.error('Error fetching latest block:', error);
        res.status(500).json({ error: 'Error fetching latest block' });
    }
});

app.listen(port, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${port}`);
});

