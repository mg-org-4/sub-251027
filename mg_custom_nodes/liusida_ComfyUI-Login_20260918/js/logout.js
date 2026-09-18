import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "Comfy.Login.Logout",
	commands: [
		{
			id: "Comfy.Login.Logout",
			label: "Logout",
			menubarLabel: "Logout", // Label shown in menu
			icon: "mdi mdi-logout", // Optional: logout icon
			tooltip: "Logout and return to login screen",
			function: async () => {
				// Save workflow data before clearing
				let workflowData = localStorage.getItem('workflow');

				// Clear all items in localStorage
				localStorage.clear();

				// Restore the workflow data
				localStorage.setItem('workflow', workflowData);

				// Clear sessionStorage if used
				sessionStorage.clear();

				// Redirect to logout endpoint
				window.location.href = "/logout";
			}
		}
	],

	menuCommands: [
		{
			path: ['File'], // Top level (or use ['File'] to put it under File menu)
			commands: ["Comfy.Login.Logout"]
		}
	],
});
