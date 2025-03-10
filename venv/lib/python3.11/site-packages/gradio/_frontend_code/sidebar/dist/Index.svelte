<script>import Sidebar from "./shared/Sidebar.svelte";
import { StatusTracker } from "@gradio/statustracker";
import Column from "@gradio/column";
export let open = true;
export let position = "left";
export let loading_status;
export let gradio;
export let width;
export let visible = true;
</script>

<StatusTracker
	autoscroll={gradio.autoscroll}
	i18n={gradio.i18n}
	{...loading_status}
/>

{#if visible}
	<Sidebar
		bind:open
		bind:position
		{width}
		on:expand={() => gradio.dispatch("expand")}
		on:collapse={() => gradio.dispatch("collapse")}
	>
		<Column>
			<slot />
		</Column>
	</Sidebar>
{/if}
