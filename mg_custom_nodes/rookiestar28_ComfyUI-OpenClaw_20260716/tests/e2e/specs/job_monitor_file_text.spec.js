import { expect, test } from '@playwright/test';
import { clickTab, mockComfyUiCore, waitForOpenClawReady } from '../utils/helpers.js';

async function installCommonRoutes(page, jobId, files) {
    await page.route(`**/history/${jobId}`, async route => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                [jobId]: {
                    status: { status_str: 'success', completed: true },
                    outputs: {
                        '9': {
                            files,
                            text: 'some generated text',
                        },
                    },
                },
            }),
        });
    });
    await page.route('**/openclaw/trace/**', async route => {
        await route.fulfill({
            status: 404,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'not_found' }),
        });
    });
}

async function addTrackedJob(page, jobId) {
    await clickTab(page, 'Jobs');
    await page.locator('input[placeholder="prompt_id"]').fill(jobId);
    await page.getByText('Add').click();
    const jobRow = page.locator('.openclaw-job-row').first();
    await expect(jobRow.locator('.openclaw-kv-val.ok')).toHaveText('completed', {
        timeout: 10000,
    });
    return jobRow;
}

test.describe('Job Monitor file-backed text output', () => {
    test.beforeEach(async ({ page }) => {
        await mockComfyUiCore(page);
        await page.goto('test-harness.html');
        await waitForOpenClawReady(page);
    });

    test('renders the official files/result.txt shape as bounded inert text', async ({ page }) => {
        const jobId = 'job-file-text-safe';
        const activeText = '<script>window.__openclawTextPwned = true</script>\n'
            + '[link](javascript:alert(1))\nSECRET_LOOKING_TEXT_TOKEN';
        const consoleMessages = [];
        let viewRequests = 0;
        page.on('console', message => consoleMessages.push(message.text()));

        await installCommonRoutes(page, jobId, [
            { filename: 'result.txt', subfolder: 'reports/2026', type: 'output' },
        ]);
        await page.route('**/api/view?*', async route => {
            viewRequests += 1;
            await route.fulfill({
                status: 200,
                contentType: 'text/plain; charset=utf-8',
                body: activeText,
            });
        });

        const jobRow = await addTrackedJob(page, jobId);
        const tile = jobRow.locator('.openclaw-job-output-text-file');
        await expect(tile).toBeVisible();
        await expect(tile.locator('.openclaw-job-output-text-content')).toHaveText(activeText);
        await expect(tile.locator('script')).toHaveCount(0);
        await expect(tile.locator('.openclaw-job-output-text-source')).toHaveAttribute(
            'href',
            /\/api\/view\?.*filename=result\.txt.*subfolder=reports%2F2026/
        );
        expect(await page.evaluate(() => window.__openclawTextPwned)).toBeUndefined();
        expect(viewRequests).toBe(1);
        expect(consoleMessages.join('\n')).not.toContain('SECRET_LOOKING_TEXT_TOKEN');
    });

    test('keeps rejected MIME content unavailable with a source link and no body leak', async ({ page }) => {
        const jobId = 'job-file-text-mime-rejected';
        const consoleMessages = [];
        page.on('console', message => consoleMessages.push(message.text()));

        await installCommonRoutes(page, jobId, [
            { filename: 'result.md', subfolder: '', type: 'output' },
        ]);
        await page.route('**/api/view?*', async route => {
            await route.fulfill({
                status: 200,
                contentType: 'text/html',
                body: '<script>SECRET_REJECTED_BODY</script>',
            });
        });

        const jobRow = await addTrackedJob(page, jobId);
        const tile = jobRow.locator('.openclaw-job-output-text-file');
        await expect(tile).toBeVisible();
        await expect(tile.locator('.openclaw-job-output-text-status')).toHaveText(
            'Text preview unavailable.'
        );
        await expect(tile.locator('.openclaw-job-output-text-source')).toBeVisible();
        await expect(tile).not.toContainText('SECRET_REJECTED_BODY');
        expect(consoleMessages.join('\n')).not.toContain('SECRET_REJECTED_BODY');
    });

    test('shows loading and a deterministic truncated preview state', async ({ page }) => {
        const jobId = 'job-file-text-truncated';
        let releaseResponse;
        const responseGate = new Promise(resolve => {
            releaseResponse = resolve;
        });

        await installCommonRoutes(page, jobId, [
            { filename: 'large.log', subfolder: '', type: 'output' },
        ]);
        await page.route('**/api/view?*', async route => {
            await responseGate;
            await route.fulfill({
                status: 200,
                contentType: 'text/plain; charset=utf-8',
                body: 'x'.repeat(5000),
            });
        });

        const jobRow = await addTrackedJob(page, jobId);
        const tile = jobRow.locator('.openclaw-job-output-text-file');
        await expect(tile.locator('.openclaw-job-output-text-status')).toHaveText(
            'Loading text preview...'
        );

        releaseResponse();
        await expect(tile.locator('.openclaw-job-output-text-status')).toHaveText(
            'Text preview truncated.'
        );
        await expect(tile.locator('.openclaw-job-output-text-content')).toHaveText(
            'x'.repeat(4096)
        );
        await expect(tile.locator('.openclaw-job-output-text-source')).toBeVisible();
    });

    test('does not let a stale text response mutate the UI after removal', async ({ page }) => {
        const jobId = 'job-file-text-stale';
        let releaseResponse;
        const responseGate = new Promise(resolve => {
            releaseResponse = resolve;
        });

        await installCommonRoutes(page, jobId, [
            { filename: 'stale.txt', subfolder: '', type: 'output' },
        ]);
        await page.route('**/api/view?*', async route => {
            await responseGate;
            try {
                await route.fulfill({
                    status: 200,
                    contentType: 'text/plain; charset=utf-8',
                    body: 'STALE_RESPONSE_MUST_NOT_RENDER',
                });
            } catch {
                // Expected when lifecycle cancellation aborts the pending request.
            }
        });

        const jobRow = await addTrackedJob(page, jobId);
        await expect(jobRow.locator('.openclaw-job-output-text-status')).toHaveText(
            'Loading text preview...'
        );
        await jobRow.getByTitle('Remove').click();
        await expect(page.locator('.openclaw-job-row')).toHaveCount(0);

        releaseResponse();
        await page.waitForTimeout(100);
        await expect(page.locator('.openclaw-job-output-text-file')).toHaveCount(0);
        await expect(page.locator('body')).not.toContainText('STALE_RESPONSE_MUST_NOT_RENDER');
    });
});
