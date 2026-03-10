# Email Notifications Setup

To receive email alerts when the monthly forecast generation fails, follow these steps:

## Enable GitHub Notifications for Issues

1. **Go to the repository**: https://github.com/conflictlab/Pace-map-risk

2. **Click "Watch"** (top right of the page)

3. **Select "Custom"** from the dropdown

4. **Enable these notifications**:
   - ✅ Issues
   - ✅ Pull requests (optional)
   - ✅ Releases (optional)

5. **Click "Apply"**

## How It Works

When the monthly forecast generation workflow fails:

1. **GitHub Action** automatically creates an issue with:
   - Title: "⚠️ Monthly Forecast Generation Failed - YYYY-MM-DD"
   - Link to failed workflow run
   - Error context and troubleshooting steps
   - Label: `automated-forecast-failure`

2. **Email Notification** is sent to you (if you enabled issue notifications above)

3. **Issue Auto-Closes** when the next successful run completes

## Check Your Email Settings

Make sure GitHub is sending you emails:

1. Go to: https://github.com/settings/notifications
2. Under "Email notification preferences":
   - ✅ Participating (issues you're involved in)
   - ✅ Watching (repositories you're watching)
3. Verify your email address is confirmed

## Testing

You can test this by:
1. Manually triggering the workflow: https://github.com/conflictlab/Pace-map-risk/actions
2. If it fails, an issue will be created
3. You should receive an email notification

## Additional Setup (Optional)

### Slack/Discord Notifications

If you want notifications in Slack or Discord instead of email, you can:

1. Add a webhook URL to your repository secrets
2. Modify `.github/workflows/monthly-forecast-generation.yml`
3. Add a notification step using the webhook

### GitHub Mobile App

Install the GitHub mobile app to receive push notifications for issues:
- iOS: https://apps.apple.com/app/github/id1477376905
- Android: https://play.google.com/store/apps/details?id=com.github.android

## Troubleshooting

**Not receiving emails?**
- Check your spam folder
- Verify notifications are enabled at https://github.com/settings/notifications
- Ensure your email is confirmed
- Check if notifications are enabled for the specific repository

**Want to test it now?**
- Go to Actions → "Generate Monthly Forecasts"
- Click "Run workflow"
- If it completes, you won't get a notification (that's good!)
- If it fails, you'll get an issue created and emailed to you
