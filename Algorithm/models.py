from django.db import models


# Create your models here.
class Videos(models.Model):
    file = models.FileField(null=True, blank=True)
    ip_link = models.TextField(null=True, blank=True)
    line_coord_init_x = models.IntegerField(null=True, blank=True)
    line_coord_init_y = models.IntegerField(null=True, blank=True)
    line_coord_end_x = models.IntegerField(null=True, blank=True)
    line_coord_end_y = models.IntegerField(null=True, blank=True)
    processed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)


class VideoLog(models.Model):
    video = models.ForeignKey('Videos', related_name='Logs', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    moving_avg = models.CharField(max_length=500, null=True, blank=True)
    data = models.CharField(max_length=500, null=True, blank=True)

# class VehicleCount(models.Model):
#     CAR = 0
#     BUS = 1
#     BIKE = 3
#     TRUCK = 4
#     THREE_WHEELER = 5
#     TRACTOR = 6
#
#     VEHICLE_CLASSES = ((CAR, "Car"), (BUS, "Bus"), (BIKE, "Bike"), (TRUCK, "Truck"), (THREE_WHEELER, "Three Wheeler"),
#                        (TRACTOR, "Tractor"))
#     INFLOW = 0
#     OUTFLOW = 1
#     FLOW_CHOICES = ((INFLOW, "Inflow"), (OUTFLOW, "Outflow"))
#     vehicle_id = models.IntegerField()
#     vehicle_type = models.IntegerField(choices=VEHICLE_CLASSES)
#     flow = models.IntegerField(choices=FLOW_CHOICES)
#     video = models.ForeignKey(Videos, related_name='logs', on_delete=models.PROTECT)
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)
