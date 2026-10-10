"""Populate the database with ~50 synthetic, already-resolved IT tickets.

Idempotent: each ticket is inserted only if a ticket with the same title AND
description does not already exist, so running it twice never duplicates data.

Run from the backend/ folder:  python seed_db.py
"""
from typing import Tuple

from database import SessionLocal, init_db
import crud

# (category, priority, title, description, resolution)
SEED_TICKETS = [
    # ---------------- Network: Wi-Fi, internet, VPN, DNS, network drive ----------------
    ("Network", "Medium", "Wi-Fi keeps disconnecting",
     "My laptop drops the office Wi-Fi every 10-15 minutes and reconnects after a while.",
     "Forget the Wi-Fi network and reconnect with your credentials. Update the wireless adapter driver from the Software Center. "
     "Turn off 'Allow the computer to turn off this device to save power' for the Wi-Fi adapter in Device Manager. "
     "If drops continue in one area, report the location to the network team to check access point coverage."),
    ("Network", "Medium", "Cannot connect to office Wi-Fi",
     "The corporate Wi-Fi rejects my password even though I can log in to my laptop.",
     "Remove the saved Wi-Fi profile and connect again. Check whether your domain password was changed recently and use the new one. "
     "Confirm the device certificate is valid in the certificate manager. Re-enroll the device in device management if the certificate has expired."),
    ("Network", "High", "No internet access but Wi-Fi connected",
     "Wi-Fi shows connected but no websites load and Teams says offline.",
     "Restart the laptop and the home router. Run the Windows network troubleshooter. "
     "Renew the IP address with ipconfig /release followed by ipconfig /renew. Check that no manual proxy is set in the proxy settings."),
    ("Network", "Low", "Slow internet speed on laptop",
     "Internet is very slow only on my laptop; my phone on the same Wi-Fi is fast.",
     "Run a speed test and compare with another device. Pause large OneDrive syncs and other bandwidth-heavy background apps. "
     "Update the network adapter driver. Run a full antivirus scan."),
    ("Network", "High", "VPN not connecting",
     "The VPN client is stuck on 'connecting' and then times out.",
     "Check that the internet connection works without the VPN. Fully quit and restart the VPN client. "
     "Confirm your password has not expired. Flush the DNS cache with ipconfig /flushdns. Reinstall the VPN client from the Software Center if it still fails."),
    ("Network", "Medium", "VPN disconnects frequently",
     "VPN drops every few minutes while I work from home.",
     "Use a wired connection or move closer to the router. Update the VPN client to the latest approved version. "
     "Switch the VPN protocol to TCP in the client settings. Turn off power saving for the network adapter."),
    ("Network", "High", "VPN authentication failed",
     "VPN rejects my username and password although I can log in to email.",
     "Reset your domain password through the self-service portal and wait 15 minutes for it to sync. "
     "Make sure the MFA prompt is approved on your phone. Ask the service desk to unlock the account if it was locked by repeated attempts."),
    ("Network", "Medium", "DNS server not responding",
     "Browser shows DNS_PROBE_FINISHED_NO_INTERNET for every website.",
     "Flush the DNS cache with ipconfig /flushdns. Make sure the adapter is set to obtain DNS servers automatically. "
     "Restart the laptop. Reset the network stack with netsh winsock reset and reboot."),
    ("Network", "Medium", "Internal websites not resolving",
     "Intranet sites say 'server not found' but public websites work.",
     "Connect to the VPN, because internal names only resolve on the corporate network. Flush the DNS cache. "
     "Check that the VPN client is using the corporate DNS servers. Try the site again in a private browser window."),
    ("Network", "High", "Network drive not accessible",
     "Mapped network drive H: shows a red X and 'path not found'.",
     "Confirm you are on the office network or connected to the VPN. Disconnect and re-map the drive with its full UNC path. "
     "Clear old saved credentials for the file server in Credential Manager. Ask the service desk to confirm you are in the folder's security group."),
    ("Network", "Medium", "Shared drive very slow",
     "Opening files from the shared network drive takes minutes.",
     "Check whether the slowness happens on VPN only or also in the office. Copy large files locally before editing. "
     "Close files left open on the share. Report the share name and time to the infrastructure team for a server health check."),
    ("Network", "Medium", "Remote desktop connection failed",
     "Remote Desktop says the remote computer cannot be reached.",
     "Make sure the remote computer is powered on and connected to the network. Connect to the VPN first. "
     "Use the full computer name rather than the short name. Ask the service desk to confirm you are in the Remote Desktop Users group."),
    # ---------------- Email ----------------
    ("Email", "High", "Outlook not opening",
     "Outlook hangs on 'Loading profile' and never opens.",
     "Start Outlook in safe mode with outlook.exe /safe. Disable recently added add-ins in File > Options > Add-ins. "
     "Run Quick Repair for Microsoft Office from Apps & features. Create a new Outlook profile if the problem remains."),
    ("Email", "High", "Outlook not syncing emails",
     "New emails appear on my phone and webmail but not in Outlook desktop.",
     "Check that Outlook is not in 'Work Offline' mode. Look at the status bar for 'Disconnected' and restart Outlook. "
     "Update Office to the latest build. Rebuild the local cache by creating a new Outlook profile."),
    ("Email", "Medium", "Emails stuck in outbox",
     "Emails stay in the Outbox and are never sent.",
     "Check the internet connection and that 'Work Offline' is turned off. Remove large attachments over the size limit and resend. "
     "Restart Outlook. Sign out of Office and sign back in to refresh the account."),
    ("Email", "Medium", "Mailbox full",
     "I get a 'mailbox is full' warning and cannot send emails.",
     "Empty the Deleted Items and Junk folders. Use Mailbox Cleanup to find large items. "
     "Move old emails to the Online Archive. Request a quota review from the service desk if the mailbox is still near the limit."),
    ("Email", "Medium", "Not receiving external emails",
     "Colleagues can email me but customers say their emails do not arrive.",
     "Check the Junk Email folder and Quarantine notifications. Review Inbox rules that may move or delete messages. "
     "Ask the sender for the exact time and address so the service desk can run a message trace."),
    ("Email", "Medium", "Outlook keeps asking for password",
     "Outlook repeatedly prompts for my password every few minutes.",
     "Sign out of all Office apps and sign back in. Remove old Office entries from Windows Credential Manager. "
     "Make sure your password was not recently changed on another device. Create a new Outlook profile if prompts continue."),
    ("Email", "Low", "Calendar not syncing on phone",
     "Meeting invites do not appear in the calendar on my phone.",
     "Open the Outlook mobile app settings and confirm calendar sync is on. Remove and re-add the account. "
     "Update the Outlook mobile app. Check that the phone is compliant in the company portal."),
    ("Email", "Low", "Shared mailbox not visible",
     "A shared mailbox I was given access to does not appear in Outlook.",
     "Wait up to 60 minutes after access is granted. Restart Outlook. "
     "Add the mailbox manually in Account Settings > More Settings > Advanced. Ask the service desk to confirm Full Access permission was applied."),
    # ---------------- Hardware: printer, laptop, audio, microphone ----------------
    ("Hardware", "Medium", "Printer not printing",
     "Documents are sent to the printer but nothing prints.",
     "Check the printer for paper, toner and error messages on its display. Clear stuck jobs from the print queue. "
     "Restart the Print Spooler service or restart the laptop. Remove and re-add the printer from the Software Center."),
    ("Hardware", "Medium", "Printer shows offline",
     "The network printer shows as offline on my laptop.",
     "Power cycle the printer and wait until it is ready. Make sure 'Use Printer Offline' is unchecked in the print queue. "
     "Confirm the laptop is on the office network. Remove and re-add the printer."),
    ("Hardware", "Low", "Printer paper jam error",
     "Printer shows a paper jam error even after removing the paper.",
     "Turn the printer off before opening any panels. Open every tray and door shown on the display and remove torn paper pieces. "
     "Check the paper is the correct size and not damp. Report to facilities if the error remains after a restart."),
    ("Hardware", "High", "Laptop not turning on",
     "The laptop does not power on even when the charger is connected.",
     "Try a different charger and wall socket. Hold the power button for 30 seconds, then connect the charger and try again. "
     "Check whether the charging light comes on. Raise a hardware repair request if there is no response."),
    ("Hardware", "Medium", "Laptop overheating",
     "My laptop gets very hot and the fan is loud all the time.",
     "Use the laptop on a hard, flat surface and keep the vents clear. Close heavy applications and browser tabs. "
     "Install pending BIOS and driver updates from the Software Center. Raise a hardware request for a cleaning check if it continues."),
    ("Hardware", "Medium", "External monitor not detected",
     "My second monitor is not detected through the docking station.",
     "Check the cable and try a different port. Press Windows+P and choose Extend. "
     "Update the graphics and docking station drivers. Test the monitor with another cable or laptop."),
    ("Hardware", "Low", "Laptop battery draining fast",
     "Battery drops from 100% to 20% within an hour.",
     "Lower screen brightness and turn on battery saver. Close apps with high usage in Task Manager. "
     "Generate a battery report with powercfg /batteryreport and share it with the service desk. Request a battery replacement if health is low."),
    ("Hardware", "Medium", "No sound from laptop speakers",
     "No audio from the laptop speakers or headphones in any application.",
     "Check the volume and that the correct output device is selected in Sound settings. Run the audio troubleshooter. "
     "Restart the Windows Audio service or restart the laptop. Update the audio driver from the Software Center."),
    ("Hardware", "Medium", "Microphone not working in Teams",
     "Other people cannot hear me in Teams calls.",
     "In Windows Privacy settings make sure microphone access is allowed for apps. Select the correct microphone in Teams > Settings > Devices. "
     "Make a test call in Teams. Unplug and reconnect the headset, and check that it is not muted on the cable."),
    ("Hardware", "Low", "Headset microphone too quiet",
     "My headset microphone works but people say my voice is very low.",
     "Open Sound settings > Input and increase the microphone level. Turn off automatic microphone adjustment in Teams if it keeps lowering the level. "
     "Try a different USB port. Test with another headset to rule out a faulty device."),
    ("Hardware", "Medium", "Webcam not working in meetings",
     "The camera is not detected in Teams or Zoom.",
     "Check Windows camera privacy settings allow app access. Close other apps that may be using the camera. "
     "Check for a physical privacy shutter or camera key. Update the camera driver and restart the laptop."),
    ("Hardware", "Low", "Keyboard keys not working",
     "Several keys on my laptop keyboard stopped responding.",
     "Restart the laptop. Check that Filter Keys is turned off in Accessibility settings. "
     "Test with an external USB keyboard. Raise a hardware repair request if the built-in keys still do not work."),
    # ---------------- Software: slow laptop, install errors, crashes, browser, disk ----------------
    ("Software", "Medium", "Laptop running slow",
     "My laptop is very slow and applications take minutes to open.",
     "Restart the laptop, especially if it has not been restarted for days. Check Task Manager for apps using high CPU or memory and close them. "
     "Disable unnecessary startup apps. Install pending Windows updates. Run a full antivirus scan."),
    ("Software", "Medium", "Slow startup after update",
     "Since the last Windows update the laptop takes 10 minutes to boot.",
     "Let the update finish fully and restart once more. Disable unnecessary startup apps in Task Manager. "
     "Check free disk space. Ask the service desk to check the device's update compliance report."),
    ("Software", "Medium", "Software installation failed",
     "Installing an application from the Software Center fails with error 0x87D00607.",
     "Restart the laptop and retry the installation. Make sure you are connected to the office network or VPN. "
     "Check that there is at least 10 GB of free disk space. Send the error code to the service desk so they can review the deployment logs."),
    ("Software", "Low", "Installer asks for admin password",
     "An installer I downloaded asks for an administrator password.",
     "Do not try to work around the prompt. Check whether the application is available in the Software Center. "
     "If not, raise a software request with business justification and manager approval so IT can package and install it."),
    ("Software", "High", "Excel crashes on startup",
     "Excel closes immediately every time I open it.",
     "Start Excel in safe mode with excel /safe. Disable COM add-ins. "
     "Run Quick Repair for Microsoft Office, then Online Repair if needed. Open a different file to check whether one workbook is the cause."),
    ("Software", "Medium", "Application not responding",
     "Our ERP desktop client freezes and shows 'Not responding'.",
     "Wait a minute in case a long operation is running. Close the app from Task Manager and reopen it. "
     "Clear the application's local cache as described in its user guide. Note the exact time and action so the application team can check logs."),
    ("Software", "Medium", "Teams not loading",
     "Microsoft Teams shows a blank white screen and never loads.",
     "Fully quit Teams from the system tray. Clear the Teams cache folder. "
     "Sign out and sign back in. Check the Microsoft 365 service status page. Reinstall Teams from the Software Center if needed."),
    ("Software", "Medium", "Browser not loading websites",
     "Chrome shows ERR_CONNECTION_RESET for some websites.",
     "Clear the browser cache and cookies. Disable recently installed extensions. "
     "Try the site in a private window or a different browser. Check the proxy settings and flush the DNS cache."),
    ("Software", "Low", "Browser very slow with many tabs",
     "Edge becomes extremely slow and uses a lot of memory.",
     "Close unused tabs and enable sleeping tabs. Remove extensions you do not need. "
     "Clear browsing data. Update the browser to the latest version."),
    ("Software", "Medium", "Disk space full",
     "C: drive is almost full and I get low disk space warnings.",
     "Run Disk Cleanup and Storage Sense to remove temporary files. Empty the Recycle Bin. "
     "Uninstall applications you no longer use. Move large personal work files to OneDrive and set them to online-only."),
    ("Software", "Medium", "Windows update stuck",
     "Windows Update is stuck at the same percentage for hours.",
     "Leave the laptop on charger for another hour. Restart and run Windows Update again. "
     "Run the Windows Update troubleshooter. Ask the service desk to push the update if it still fails."),
    ("Software", "Low", "Excel file corrupted",
     "Excel says the file is corrupt and cannot be opened.",
     "Use File > Open, select the file, and choose Open and Repair. Restore an earlier version from OneDrive or SharePoint version history. "
     "Ask the service desk to restore the file from backup if no good version exists."),
    # ---------------- Access: passwords, lockouts, MFA, permissions ----------------
    ("Access", "High", "Forgot password",
     "I forgot my Windows password and cannot log in to my laptop.",
     "Use the self-service password reset portal from another device. If self-service is not set up, contact the service desk to verify your identity. "
     "Log in with the new password while connected to the office network or VPN so the laptop updates its cached password."),
    ("Access", "High", "Account locked out",
     "My account is locked after several wrong password attempts.",
     "Wait 30 minutes for the automatic unlock or contact the service desk to verify your identity and unlock the account. "
     "Update the password saved on your phone's email app and other devices, because old saved passwords cause repeated lockouts."),
    ("Access", "Medium", "Password expired",
     "I get a message that my password has expired when I log in.",
     "Change the password using Ctrl+Alt+Del > Change a password while on the office network or VPN. "
     "Follow the password policy for length and complexity. Update the new password on your mobile devices."),
    ("Access", "High", "MFA code not received",
     "I am not getting the authenticator push or SMS code when I sign in.",
     "Check that your phone has network signal and the time is set automatically. Open the authenticator app and pull to refresh. "
     "Use an alternative verification method if one is registered. Contact the service desk to re-register MFA after identity verification."),
    ("Access", "Medium", "New phone, MFA not working",
     "I changed my phone and the authenticator app no longer works for login.",
     "Sign in to the security info page from a trusted device and add the new phone. "
     "If you cannot sign in at all, contact the service desk; they will verify your identity and reset your MFA registration."),
    ("Access", "Medium", "Single sign-on login loop",
     "Logging in to a web application keeps redirecting me back to the login page.",
     "Clear the browser cookies for the sign-in site. Try a private browser window. "
     "Make sure the laptop date and time are correct. Ask the service desk to confirm you are assigned to the application."),
    ("Access", "Medium", "Permission denied on shared folder",
     "I get 'Access denied' when opening a project folder on the file server.",
     "Ask the folder owner or your manager to approve access. Raise an access request naming the exact folder path. "
     "After access is granted, sign out and back in so your permissions refresh."),
    ("Access", "Low", "Need access to SharePoint site",
     "I need access to the Finance team SharePoint site for a new project.",
     "Click 'Request access' on the SharePoint page so the site owner is notified. Alternatively raise an access request with manager approval. "
     "Access usually applies within 15 minutes of approval."),
]


def seed(session=None) -> Tuple[int, int]:
    """Insert missing seed tickets. Returns (inserted, skipped)."""
    init_db()
    own_session = session is None
    db = session or SessionLocal()
    inserted = skipped = 0
    try:
        for category, priority, title, description, resolution in SEED_TICKETS:
            if crud.ticket_exists(db, title, description):
                skipped += 1
                continue
            crud.create_ticket(db, title=title, description=description, category=category,
                               priority=priority, resolution=resolution)
            inserted += 1
    finally:
        if own_session:
            db.close()
    return inserted, skipped


if __name__ == "__main__":
    added, existing = seed()
    print(f"Seed complete: {added} inserted, {existing} already present "
          f"({len(SEED_TICKETS)} seed tickets total).")
